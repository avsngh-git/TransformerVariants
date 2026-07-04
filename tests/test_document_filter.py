"""Tests for DocumentFilter component."""

import pytest

from src.data.streaming_prepare import DocumentFilter, FilterStats


class TestDocumentFilterInit:
    """Tests for DocumentFilter initialization."""

    def test_default_thresholds(self):
        """DocumentFilter uses PipelineConfig defaults (50, 10000)."""
        f = DocumentFilter()
        assert f._min_tokens == 50
        assert f._max_tokens == 10000

    def test_custom_thresholds(self):
        """DocumentFilter accepts custom min/max values."""
        f = DocumentFilter(min_tokens=100, max_tokens=5000)
        assert f._min_tokens == 100
        assert f._max_tokens == 5000

    def test_initial_stats_are_zero(self):
        """All stats counters start at zero."""
        f = DocumentFilter()
        stats = f.stats
        assert stats.documents_processed == 0
        assert stats.documents_accepted == 0
        assert stats.documents_filtered_short == 0
        assert stats.documents_filtered_long == 0


class TestDocumentFilterShouldAccept:
    """Tests for should_accept method."""

    def test_rejects_below_min_tokens(self):
        """Documents with token_count < min_tokens are rejected."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        assert f.should_accept(49) is False
        assert f.should_accept(1) is False
        assert f.should_accept(0) is False

    def test_accepts_at_min_tokens(self):
        """Documents with exactly min_tokens are accepted."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        assert f.should_accept(50) is True

    def test_accepts_between_min_and_max(self):
        """Documents between min and max are accepted."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        assert f.should_accept(500) is True
        assert f.should_accept(5000) is True

    def test_accepts_at_max_tokens(self):
        """Documents with exactly max_tokens are accepted (no truncation needed)."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        assert f.should_accept(10000) is True

    def test_accepts_above_max_tokens(self):
        """Documents above max_tokens are still accepted (truncated later)."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        assert f.should_accept(10001) is True
        assert f.should_accept(50000) is True


class TestDocumentFilterStats:
    """Tests for filter statistics tracking."""

    def test_documents_processed_increments_on_every_call(self):
        """documents_processed counts every call to should_accept."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10)   # rejected
        f.should_accept(100)  # accepted
        f.should_accept(20000)  # accepted (long)
        assert f.stats.documents_processed == 3

    def test_documents_accepted_counts_accepted(self):
        """documents_accepted only counts accepted documents."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10)   # rejected
        f.should_accept(100)  # accepted
        f.should_accept(20000)  # accepted (long)
        assert f.stats.documents_accepted == 2

    def test_filtered_short_counts_rejections(self):
        """documents_filtered_short counts short document rejections."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10)
        f.should_accept(49)
        f.should_accept(100)
        assert f.stats.documents_filtered_short == 2

    def test_filtered_long_counts_over_max(self):
        """documents_filtered_long counts accepted docs exceeding max."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10001)
        f.should_accept(50000)
        f.should_accept(100)
        assert f.stats.documents_filtered_long == 2

    def test_exactly_max_not_counted_as_long(self):
        """Documents with exactly max_tokens are NOT counted as filtered_long."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10000)
        assert f.stats.documents_filtered_long == 0
        assert f.stats.documents_accepted == 1

    def test_stats_consistency(self):
        """processed == accepted + filtered_short always holds."""
        f = DocumentFilter(min_tokens=50, max_tokens=10000)
        f.should_accept(10)     # short
        f.should_accept(100)    # normal
        f.should_accept(20000)  # long (accepted)
        f.should_accept(49)     # short
        f.should_accept(5000)   # normal

        stats = f.stats
        assert stats.documents_processed == 5
        assert stats.documents_accepted == 3
        assert stats.documents_filtered_short == 2
        assert stats.documents_filtered_long == 1
        # Invariant: processed = accepted + filtered_short
        assert stats.documents_processed == stats.documents_accepted + stats.documents_filtered_short


class TestDocumentFilterTruncate:
    """Tests for the truncate method."""

    def test_no_truncation_when_within_max(self):
        """Tokens within max_tokens are returned unchanged."""
        f = DocumentFilter(min_tokens=50, max_tokens=100)
        tokens = list(range(80))
        result = f.truncate(tokens, 80)
        assert result == tokens
        assert len(result) == 80

    def test_no_truncation_at_exactly_max(self):
        """Tokens at exactly max_tokens are returned unchanged."""
        f = DocumentFilter(min_tokens=50, max_tokens=100)
        tokens = list(range(100))
        result = f.truncate(tokens, 100)
        assert result == tokens
        assert len(result) == 100

    def test_truncation_when_above_max(self):
        """Tokens above max_tokens are truncated to max_tokens."""
        f = DocumentFilter(min_tokens=50, max_tokens=100)
        tokens = list(range(150))
        result = f.truncate(tokens, 150)
        assert len(result) == 100
        assert result == list(range(100))

    def test_truncation_preserves_order(self):
        """Truncation takes the first max_tokens tokens."""
        f = DocumentFilter(min_tokens=10, max_tokens=5)
        tokens = [10, 20, 30, 40, 50, 60, 70]
        result = f.truncate(tokens, 7)
        assert result == [10, 20, 30, 40, 50]

    def test_truncate_does_not_affect_stats(self):
        """Calling truncate does not modify filter stats."""
        f = DocumentFilter(min_tokens=50, max_tokens=100)
        f.truncate(list(range(150)), 150)
        assert f.stats.documents_processed == 0


class TestDocumentFilterStatsProperty:
    """Tests for the stats property."""

    def test_stats_returns_filter_stats_instance(self):
        """stats property returns a FilterStats instance."""
        f = DocumentFilter()
        assert isinstance(f.stats, FilterStats)

    def test_stats_reflects_live_state(self):
        """stats property reflects the current state after operations."""
        f = DocumentFilter(min_tokens=10, max_tokens=100)
        assert f.stats.documents_processed == 0
        f.should_accept(5)
        assert f.stats.documents_processed == 1
        assert f.stats.documents_filtered_short == 1
