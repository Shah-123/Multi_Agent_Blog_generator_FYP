"""
Real-Time Event Bus for Agent Visualization
Lightweight pub/sub system using asyncio.Queue.
Agent nodes push events → WebSocket endpoint streams them to the frontend.
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict

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
_event_history: Dict[str, List[dict]] = defaultdict(list)


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
    _event_history[job_id].append(event_dict)
    
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
        # We need to find the main event loop to enqueue thread-safely
        loop = asyncio.get_running_loop()
        loop.call_soon_threadsafe(_enqueue_safely)
    except RuntimeError:
        # If get_running_loop() fails (e.g. we are the main thread somehow, or no loop)
        _enqueue_safely()


def subscribe(job_id: str) -> asyncio.Queue:
    """
    Subscribe to events for a job.
    Returns a Queue that will receive future events.
    Also replays any past events.
    """
    queue = asyncio.Queue(maxsize=200)

    # Replay history for late joiners
    for past_event in _event_history.get(job_id, []):
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
        # Clean up if no subscribers left
        if not _subscribers[job_id]:
            del _subscribers[job_id]


def clear_job(job_id: str):
    """Clean up all data for a completed/failed job."""
    _subscribers.pop(job_id, None)
    # Keep history for 5 mins, then let it GC
    # (In production you'd use a TTL cache)


def get_history(job_id: str) -> List[dict]:
    """Get all past events for a job."""
    return list(_event_history.get(job_id, []))
