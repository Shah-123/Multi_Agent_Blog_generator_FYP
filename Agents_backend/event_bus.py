"""
Real-Time Event Bus for Agent Visualization
Lightweight pub/sub system using asyncio.Queue.
Agent nodes push events → WebSocket endpoint streams them to the frontend.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger("event_bus")

# ============================================================================
# EVENT DATA MODEL
# ============================================================================

@dataclass
class AgentEvent:
    """A single event emitted by an agent node."""
    job_id: str
    agent_name: str
    status: str          # "started", "working", "completed", "error"
    message: str
    timestamp: float
    metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["metrics"] is None:
            d["metrics"] = {}
        return d


# ============================================================================
# GLOBAL EVENT BUS
# ============================================================================

# Stores: job_id -> list of subscriber queues
_subscribers: Dict[str, List[asyncio.Queue]] = defaultdict(list)

# Stores: job_id -> list of past events (for late-joining clients)
# Each entry is (timestamp, event_dict) so TTL cleanup can filter by age.
_event_history: Dict[str, List[tuple]] = defaultdict(list)

# ✅ NEW: How long (seconds) to retain history for a completed job.
# 5 minutes is enough for any UI to catch up; after that, memory is freed.
_HISTORY_TTL_SECONDS = 300  # 5 minutes

# ✅ NEW: How often the cleanup task runs (seconds).
_CLEANUP_INTERVAL_SECONDS = 60  # every 1 minute

# ✅ NEW: Reference to the background cleanup task so it can be cancelled on shutdown.
_cleanup_task: Optional[asyncio.Task] = None


# ============================================================================
# TTL CLEANUP
# ============================================================================

async def _cleanup_loop():
    """
    Background coroutine that periodically removes stale job history.

    Runs every _CLEANUP_INTERVAL_SECONDS. For each job, drops events whose
    timestamp is older than _HISTORY_TTL_SECONDS. If a job's history becomes
    empty AND it has no active subscribers, the job entry is deleted entirely.

    This prevents _event_history from growing unboundedly across many runs —
    the original code noted this as a known limitation but never fixed it.
    """
    while True:
        try:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            _run_cleanup()
        except asyncio.CancelledError:
            # Graceful shutdown — exit the loop cleanly.
            logger.info("Event bus cleanup task cancelled.")
            break
        except Exception as e:
            # Never let the cleanup task crash silently; log and keep running.
            logger.warning(f"Event bus cleanup error (non-fatal): {e}")


def _run_cleanup():
    """
    Synchronous cleanup logic, separated from the async loop so it can also
    be called directly in tests without needing a running event loop.
    """
    cutoff = time.time() - _HISTORY_TTL_SECONDS
    stale_jobs = []

    for job_id, history in list(_event_history.items()):
        # Each entry is (timestamp, event_dict) — filter out old ones.
        fresh = [(ts, ev) for (ts, ev) in history if ts >= cutoff]

        if not fresh and not _subscribers.get(job_id):
            # No recent events AND no active subscribers — safe to delete.
            stale_jobs.append(job_id)
        else:
            _event_history[job_id] = fresh

    for job_id in stale_jobs:
        _event_history.pop(job_id, None)
        _subscribers.pop(job_id, None)

    if stale_jobs:
        logger.info(f"Event bus: cleaned up {len(stale_jobs)} stale job(s): {stale_jobs}")


def start_cleanup_task() -> asyncio.Task:
    """
    Start the background TTL cleanup coroutine.

    Call this once from your FastAPI startup event or from an async context.
    Stores the task reference so it can be cancelled on shutdown.

    Example (FastAPI):
        @app.on_event("startup")
        async def startup():
            event_bus.start_cleanup_task()

        @app.on_event("shutdown")
        async def shutdown():
            event_bus.stop_cleanup_task()
    """
    global _cleanup_task
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.ensure_future(_cleanup_loop())
        logger.info("Event bus cleanup task started.")
    return _cleanup_task


def stop_cleanup_task():
    """Cancel the background cleanup task gracefully."""
    global _cleanup_task
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        _cleanup_task = None


# ============================================================================
# CORE EMIT / SUBSCRIBE API  (unchanged interface)
# ============================================================================

def emit(job_id: str, agent_name: str, status: str, message: str, metrics: dict = None):
    """
    Emit an event from an agent node.
    Called synchronously from LangGraph nodes which run in a BackgroundTask thread.
    Uses call_soon_threadsafe to safely bridge back to the main asyncio loop.
    """
    if not job_id:
        return

    event = AgentEvent(
        job_id=job_id,
        agent_name=agent_name,
        status=status,
        message=message,
        timestamp=time.time(),
        metrics=metrics or {},
    )

    event_dict = event.to_dict()

    # ✅ Store as (timestamp, event_dict) for TTL-aware cleanup.
    _event_history[job_id].append((event.timestamp, event_dict))

    queues = _subscribers.get(job_id, [])
    if not queues:
        return

    def _enqueue_safely():
        for q in queues:
            try:
                q.put_nowait(event_dict)
            except asyncio.QueueFull:
                pass

    try:
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_enqueue_safely)
    except RuntimeError:
        _enqueue_safely()


def subscribe(job_id: str) -> asyncio.Queue:
    """
    Subscribe to events for a job.
    Returns a Queue that will receive future events.
    Also replays any past events (TTL-filtered).
    """
    queue = asyncio.Queue(maxsize=200)

    # Replay history — unwrap (timestamp, event_dict) tuples.
    for _ts, past_event in _event_history.get(job_id, []):
        try:
            queue.put_nowait(past_event)
        except asyncio.QueueFull:
            break

    _subscribers[job_id].append(queue)
    return queue


def unsubscribe(job_id: str, queue: asyncio.Queue):
    """Remove a subscriber queue."""
    if job_id in _subscribers:
        try:
            _subscribers[job_id].remove(queue)
        except ValueError:
            pass
        if not _subscribers[job_id]:
            del _subscribers[job_id]


def clear_job(job_id: str):
    """Immediately clean up all data for a completed/failed job."""
    _subscribers.pop(job_id, None)
    _event_history.pop(job_id, None)


def get_history(job_id: str) -> List[dict]:
    """
    Get all stored events for a job (TTL-filtered).
    Returns plain event dicts (timestamps are internal only).
    """
    return [ev for _ts, ev in _event_history.get(job_id, [])]