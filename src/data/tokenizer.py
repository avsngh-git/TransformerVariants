"""Tokenizer utilities for the data pipeline.

Thin wrapper around tiktoken that isolates the tokenizer choice
from the rest of the codebase. No other module should import tiktoken
directly — they go through these functions instead.
"""

import tiktoken


def get_tokenizer(name: str = "gpt2") -> tiktoken.Encoding:
    """Load a tiktoken encoding by name.

    Args:
        name: The encoding name. Supported: "gpt2", "cl100k_base", "o200k_base".
              Defaults to "gpt2" (50,257 vocab, used by GPT-2/GPT-3).

    Returns:
        A tiktoken Encoding object that can encode/decode text.

    Raises:
        ValueError: If the encoding name is not recognized by tiktoken.
    """
    try:
        return tiktoken.get_encoding(name)
    except ValueError as e:
        available = tiktoken.list_encoding_names()
        raise ValueError(
            f"Unknown tokenizer '{name}'. Available: {available}"
        ) from e


def encode(text: str, tokenizer: tiktoken.Encoding) -> list[int]:
    """Encode a string into a list of integer token IDs.

    Args:
        text: The raw text to tokenize.
        tokenizer: A tiktoken Encoding object (from get_tokenizer).

    Returns:
        A list of integer token IDs. Length depends on the text and
        the tokenizer's vocabulary/merge rules.
    """
    return tokenizer.encode(text, allowed_special={"<|endoftext|>"})


def decode(tokens: list[int], tokenizer: tiktoken.Encoding) -> str:
    """Decode a list of token IDs back into a string.

    Args:
        tokens: A list of integer token IDs.
        tokenizer: A tiktoken Encoding object (from get_tokenizer).

    Returns:
        The decoded string. Note that encode → decode is lossless for
        valid token sequences, but decode → encode → decode may differ
        if the original text had unusual whitespace or special characters.
    """
    return tokenizer.decode(tokens)


def get_eot_token(tokenizer: tiktoken.Encoding) -> int:
    """Get the end-of-text (EOT) token ID for this tokenizer.

    The EOT token is inserted between documents when concatenating them
    into a single token stream. It signals to the model that one document
    has ended and another begins.

    Args:
        tokenizer: A tiktoken Encoding object (from get_tokenizer).

    Returns:
        The integer token ID for the EOT token.
        For GPT-2 this is 50256 (the last token in the vocabulary).
    """
    return tokenizer.eot_token


def get_vocab_size(tokenizer: tiktoken.Encoding) -> int:
    """Get the total vocabulary size for this tokenizer.

    This determines the size of the model's embedding layer and is
    recorded in the manifest so the training code knows what to expect.

    Args:
        tokenizer: A tiktoken Encoding object (from get_tokenizer).

    Returns:
        The number of tokens in the vocabulary (including special tokens).
        For GPT-2 this is 50257.
    """
    return tokenizer.n_vocab
