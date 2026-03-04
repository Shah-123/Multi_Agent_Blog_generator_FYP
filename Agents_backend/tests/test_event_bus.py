"""
Tests for event_bus.py — real-time event pub/sub system.

Tests the synchronous, non-asyncio parts of the event bus:
emit, get_history, clear_job, _run_cleanup, and AgentEvent.
"""
import time
import pytest
from event_bus import (
    AgentEvent,
    emit,
    get_history,
    clear_job,
    _run_cleanup,
    _event_history,
    _subscribers,
    _HISTORY_TTL_SECONDS,
)


@pytest.fixture(autouse=True)
def clean_event_bus():
    """Ensure each test starts with a clean event bus."""
    _event_history.clear()
    _subscribers.clear()
    yield
    _event_history.clear()
    _subscribers.clear()


# ========================================================================
# AgentEvent
# ========================================================================

class TestAgentEvent:
    """Tests for the AgentEvent dataclass."""

    def test_to_dict(self):
        event = AgentEvent(
            job_id="job1", agent_name="router", status="started",
            message="Analyzing topic", timestamp=1000.0
        )
        d = event.to_dict()
        assert d["job_id"] == "job1"
        assert d["agent_name"] == "router"
        assert d["metrics"] == {}  # None → {}

    def test_to_dict_with_metrics(self):
        event = AgentEvent(
            job_id="job1", agent_name="qa", status="completed",
            message="Done", timestamp=1000.0, metrics={"score": 8.5}
        )
        d = event.to_dict()
        assert d["metrics"] == {"score": 8.5}


# ========================================================================
# emit + get_history
# ========================================================================

class TestEmitAndHistory:
    """Tests for emit() and get_history()."""

    def test_emit_stores_event_in_history(self):
        emit("job_test", "router", "started", "Starting analysis")
        history = get_history("job_test")
        assert len(history) == 1
        assert history[0]["agent_name"] == "router"
        assert history[0]["message"] == "Starting analysis"

    def test_multiple_emits_accumulate(self):
        emit("job_test", "router", "started", "Step 1")
        emit("job_test", "researcher", "started", "Step 2")
        emit("job_test", "qa", "completed", "Step 3")
        history = get_history("job_test")
        assert len(history) == 3

    def test_separate_jobs_have_separate_history(self):
        emit("job_a", "router", "started", "Job A event")
        emit("job_b", "router", "started", "Job B event")
        assert len(get_history("job_a")) == 1
        assert len(get_history("job_b")) == 1

    def test_empty_job_id_is_skipped(self):
        emit("", "router", "started", "Should be skipped")
        assert get_history("") == []


# ========================================================================
# clear_job
# ========================================================================

class TestClearJob:
    """Tests for clear_job()."""

    def test_clears_history(self):
        emit("job_clear", "agent", "started", "Something")
        assert len(get_history("job_clear")) == 1
        clear_job("job_clear")
        assert get_history("job_clear") == []

    def test_no_error_on_nonexistent_job(self):
        clear_job("nonexistent_job")  # should not raise


# ========================================================================
# _run_cleanup
# ========================================================================

class TestRunCleanup:
    """Tests for the synchronous _run_cleanup() function."""

    def test_removes_stale_events(self):
        # Inject a fake event with a very old timestamp
        old_ts = time.time() - _HISTORY_TTL_SECONDS - 100
        _event_history["old_job"].append((old_ts, {"agent_name": "test", "status": "done"}))
        _run_cleanup()
        assert "old_job" not in _event_history

    def test_keeps_fresh_events(self):
        emit("fresh_job", "agent", "started", "Recent event")
        _run_cleanup()
        assert len(get_history("fresh_job")) == 1
