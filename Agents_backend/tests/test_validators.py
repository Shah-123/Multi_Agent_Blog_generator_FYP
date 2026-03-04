"""
Tests for validators.py — TopicValidator._basic_syntax_check.

Only the fast, free syntax check is tested (no LLM calls needed).
The LLM gatekeeper is tested implicitly during integration.
"""
import pytest
from validators import TopicValidator


@pytest.fixture
def validator():
    return TopicValidator()


class TestBasicSyntaxCheck:
    """Tests for TopicValidator._basic_syntax_check()."""

    def test_accepts_valid_topic(self, validator):
        valid, reason = validator._basic_syntax_check("The Future of AI in Healthcare")
        assert valid is True

    def test_rejects_empty_string(self, validator):
        valid, reason = validator._basic_syntax_check("")
        assert valid is False
        assert "empty" in reason.lower() or "short" in reason.lower()

    def test_rejects_whitespace_only(self, validator):
        valid, reason = validator._basic_syntax_check("     ")
        assert valid is False

    def test_rejects_very_short_topic(self, validator):
        valid, reason = validator._basic_syntax_check("Hi")
        assert valid is False

    def test_rejects_very_long_topic(self, validator):
        long_topic = "A" * 300
        valid, reason = validator._basic_syntax_check(long_topic)
        assert valid is False
        assert "long" in reason.lower() or "characters" in reason.lower()

    def test_accepts_moderately_long_topic(self, validator):
        """A 50-character topic should be fine."""
        valid, reason = validator._basic_syntax_check("How Blockchain Is Changing Financial Services Now")
        assert valid is True
