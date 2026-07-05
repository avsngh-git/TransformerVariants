"""Mixture of Experts Feed-Forward Network.

Implements Mixtral-style top-k expert routing with SwiGLU FFN experts.
Each MoE layer contains N independent SwiGLU expert FFNs and a learned
Router that selects the top-k experts per token.

The output for each token is a weighted sum of the selected experts' outputs,
using renormalized router probabilities as weights.

Auxiliary losses (load-balancing + z-loss) prevent expert collapse during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import ModelConfig
from src.models.swiglu_ffn import SwiGLUFeedForward


class MoEFeedForward(nn.Module):
    """Mixture of Experts feed-forward layer with top-k routing.

    Drop-in replacement for SwiGLUFeedForward when num_experts is configured.
    Same input/output contract: (batch, seq_len, d_model) → (batch, seq_len, d_model).

    Args:
        config: ModelConfig with num_experts, moe_top_k, aux_loss_alpha,
                z_loss_beta, d_model, bias, dropout fields.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts: int = config.num_experts
        self.top_k: int = config.moe_top_k
        self.aux_loss_alpha: float = config.aux_loss_alpha
        self.z_loss_beta: float = config.z_loss_beta

        # Validation
        if self.top_k < 1 or self.top_k > self.num_experts:
            raise ValueError(
                f"moe_top_k must be between 1 and num_experts ({self.num_experts}) "
                f"inclusive, got {self.top_k}"
            )

        # Router: learned linear projection (no bias, as per Mixtral)
        self.router = nn.Linear(config.d_model, self.num_experts, bias=False)

        # Expert FFNs: N independent SwiGLU instances
        self.experts = nn.ModuleList([
            SwiGLUFeedForward(config) for _ in range(self.num_experts)
        ])

        # Internal aux loss buffer (set during forward, cleared on retrieval)
        self._aux_loss: torch.Tensor | None = None

        # Routing data capture
        self.record_routing: bool = False
        self._routing_buffer: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Route tokens to top-k experts, return weighted combination.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model).
        """
        B, T, D = x.shape

        # 1. Compute router logits and probabilities
        router_logits = self.router(x)                    # (B, T, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)   # (B, T, num_experts)

        # 2. Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # both (B, T, top_k)

        # 3. Renormalize selected probabilities
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 4. Compute expert outputs (loop over experts)
        # Flatten batch and sequence for routing
        x_flat = x.view(B * T, D)                        # (B*T, D)
        output = torch.zeros_like(x_flat)                 # (B*T, D)
        flat_indices = top_k_indices.view(B * T, self.top_k)
        flat_weights = top_k_weights.view(B * T, self.top_k)

        for expert_idx in range(self.num_experts):
            # Find which tokens selected this expert (across all top-k slots)
            # expert_mask: (B*T, top_k) boolean
            expert_mask = (flat_indices == expert_idx)
            # token_mask: (B*T,) boolean — True if any top-k slot selected this expert
            token_mask = expert_mask.any(dim=-1)

            if not token_mask.any():
                continue

            # Gather tokens for this expert
            expert_input = x_flat[token_mask]             # (num_selected, D)
            expert_output = self.experts[expert_idx](
                expert_input.unsqueeze(0)                  # (1, num_selected, D)
            ).squeeze(0)                                   # (num_selected, D)

            # Weighted contribution: sum weights across all top-k slots that chose this expert
            # weight_for_tokens: (num_selected,)
            weight_for_tokens = (flat_weights * expert_mask.float()).sum(dim=-1)[token_mask]

            output[token_mask] += weight_for_tokens.unsqueeze(-1) * expert_output

        output = output.view(B, T, D)

        # 5. Compute and store auxiliary losses (training only)
        if self.training:
            self._aux_loss = self._compute_aux_loss(router_logits, router_probs, top_k_indices)

        # 6. Record routing data if enabled
        if self.record_routing:
            self._routing_buffer.append((
                top_k_indices.detach(),   # (B, T, top_k)
                top_k_weights.detach(),   # (B, T, top_k)
            ))

        return output

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,   # (B, T, num_experts)
        router_probs: torch.Tensor,    # (B, T, num_experts)
        top_k_indices: torch.Tensor,   # (B, T, top_k)
    ) -> torch.Tensor:
        """Compute combined load-balancing and z-loss.

        Aux loss: α * num_experts * Σ(f_i * P_i)
          - f_i = fraction of tokens routed to expert i
          - P_i = mean router probability for expert i

        Z-loss: β * mean(log(Σ exp(router_logits))²)

        Returns:
            Scalar tensor that participates in autograd.
        """
        num_tokens = router_logits.shape[0] * router_logits.shape[1]

        # --- Load-balancing aux loss ---
        aux_loss = torch.tensor(0.0, device=router_logits.device)
        if self.aux_loss_alpha > 0.0:
            # f_i: fraction of tokens assigned to each expert
            # Count across all top-k slots
            one_hot = F.one_hot(top_k_indices, self.num_experts).float()  # (B, T, top_k, E)
            tokens_per_expert = one_hot.sum(dim=(0, 1, 2))  # (E,)
            f = tokens_per_expert / (num_tokens * self.top_k)  # normalize by total assignments

            # P_i: mean router probability per expert
            P = router_probs.mean(dim=(0, 1))  # (E,)

            aux_loss = self.aux_loss_alpha * (f * P).sum() * self.num_experts

        # --- Z-loss ---
        z_loss = torch.tensor(0.0, device=router_logits.device)
        if self.z_loss_beta > 0.0:
            # log(Σ exp(logits))² — the logsumexp squared
            log_z = torch.logsumexp(router_logits, dim=-1)  # (B, T)
            z_loss = self.z_loss_beta * (log_z ** 2).mean()

        return aux_loss + z_loss

    def get_aux_loss(self) -> torch.Tensor:
        """Return stored auxiliary loss and clear the buffer.

        Returns:
            Scalar tensor (participates in autograd). Zero if no forward
            pass has occurred or if alpha and beta are both 0.
        """
        if self._aux_loss is None:
            return torch.tensor(0.0)
        loss = self._aux_loss
        self._aux_loss = None
        return loss

    def get_routing_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return and clear the routing data buffer.

        Returns:
            List of (expert_indices, expert_weights) tuples from buffered
            forward passes. Each expert_indices has shape (batch, seq_len, top_k),
            expert_weights has shape (batch, seq_len, top_k).
        """
        data = self._routing_buffer
        self._routing_buffer = []
        return data
