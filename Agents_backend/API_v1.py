import os
import json
import uuid
import asyncio
import sqlite3
import secrets
import time
import re as re_module
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Security, Depends, WebSocket, WebSocketDisconnect
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

from main import create_blog_structure, save_blog_content, generate_readme, build_graph
from Graph.state import State
from Graph.templates import TOPIC_SUGGESTIONS_SYSTEM
from validators import TopicValidator
from event_bus import subscribe, unsubscribe, emit as _emit, get_history

# ============================================================================
# AUTH
# ============================================================================

API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def _get_expected_key() -> str:
    key = os.getenv("API_SECRET_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "API_SECRET_KEY is not set in your .env file. "
            "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    return key

async def require_api_key(api_key: str = Security(_api_key_header)):
    """Dependency — inject into any endpoint that should be protected."""
    if not api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header is missing")
    if not secrets.compare_digest(api_key, _get_expected_key()):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# ============================================================================
# RATE LIMITING (in-memory, per API key)
# ============================================================================

_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))  # max requests/min
_rate_store: Dict[str, list] = defaultdict(list)

def _check_rate_limit(api_key: str):
    """Raises 429 if the key has exceeded the rate limit."""
    now = time.time()
    window = [t for t in _rate_store[api_key] if now - t < 60]
    _rate_store[api_key] = window
    if len(window) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded ({_RATE_LIMIT} requests/min). Try again later.",
            headers={"Retry-After": "60"},
        )
    _rate_store[api_key].append(now)

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AI Content Factory API",
    description="Generate blogs, social media content, and podcasts with AI",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Restrict CORS to your actual frontend origin in production.
_allowed_origins = [
    os.getenv("ALLOWED_ORIGIN", "http://localhost:3000"),
    "http://localhost:8000",  # Same-origin for embedded frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# ============================================================================
# SQLITE JOB MANAGER
# ============================================================================

DB_PATH = "jobs.db"

def _init_db():
    """Create the jobs table if it doesn't exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id          TEXT PRIMARY KEY,
                topic       TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'pending',
                progress    TEXT NOT NULL DEFAULT '{}',
                settings    TEXT NOT NULL DEFAULT '{}',
                plan        TEXT,
                result      TEXT,
                error       TEXT,
                created_at  TEXT NOT NULL,
                started_at  TEXT,
                completed_at TEXT
            )
        """)
        conn.commit()

_init_db()

@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def _default_progress() -> dict:
    return {
        "router": False, "research": False, "planning": False,
        "writing": False, "images": False, "fact_check": False,
        "social_media": False, "podcast": False, "evaluation": False
    }

class SQLiteJobManager:
    """
    Persistent job store backed by SQLite.
    All JSON fields are serialised on write and deserialised on read.
    """

    # Keep MemorySaver instances in RAM — they hold the LangGraph checkpoint
    # that makes the HITL resume possible. They are intentionally transient;
    # only the job metadata needs to survive a restart.
    _memory_store: Dict[str, MemorySaver] = {}

    # ------------------------------------------------------------------ write
    def create_job(self, topic: str, settings: dict) -> str:
        job_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with _db() as conn:
            conn.execute(
                """INSERT INTO jobs
                   (id, topic, status, progress, settings, created_at)
                   VALUES (?, ?, 'pending', ?, ?, ?)""",
                (job_id, topic,
                 json.dumps(_default_progress()),
                 json.dumps(settings),
                 now)
            )
        self._memory_store[job_id] = MemorySaver()
        return job_id

    def update_progress(self, job_id: str, stage: str):
        job = self.get_job(job_id)
        if not job:
            return
        progress = job["progress"]
        progress[stage] = True
        with _db() as conn:
            conn.execute(
                "UPDATE jobs SET progress = ? WHERE id = ?",
                (json.dumps(progress), job_id)
            )

    def start_job(self, job_id: str):
        with _db() as conn:
            conn.execute(
                "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
                (datetime.now().isoformat(), job_id)
            )

    def store_plan(self, job_id: str, plan_dict: dict):
        """Persist the generated plan so the /approve endpoint can read it."""
        with _db() as conn:
            conn.execute(
                "UPDATE jobs SET plan = ?, status = 'awaiting_approval' WHERE id = ?",
                (json.dumps(plan_dict), job_id)
            )

    def complete_job(self, job_id: str, result: dict):
        with _db() as conn:
            conn.execute(
                """UPDATE jobs
                   SET status = 'completed', result = ?, completed_at = ?
                   WHERE id = ?""",
                (json.dumps(result), datetime.now().isoformat(), job_id)
            )

    def fail_job(self, job_id: str, error: str):
        with _db() as conn:
            conn.execute(
                """UPDATE jobs
                   SET status = 'failed', error = ?, completed_at = ?
                   WHERE id = ?""",
                (error, datetime.now().isoformat(), job_id)
            )

    def delete_job(self, job_id: str):
        with _db() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        self._memory_store.pop(job_id, None)

    # ------------------------------------------------------------------ read
    def get_job(self, job_id: str) -> Optional[dict]:
        with _db() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["progress"] = json.loads(d["progress"])
        d["settings"] = json.loads(d["settings"])
        d["plan"]     = json.loads(d["plan"])   if d["plan"]   else None
        d["result"]   = json.loads(d["result"]) if d["result"] else None
        return d

    def list_jobs(self, limit: int = 50) -> List[dict]:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        jobs = []
        for row in rows:
            d = dict(row)
            d["progress"] = json.loads(d["progress"])
            d["settings"] = json.loads(d["settings"])
            d["plan"]     = json.loads(d["plan"])   if d["plan"]   else None
            d["result"]   = json.loads(d["result"]) if d["result"] else None
            jobs.append(d)
        return jobs

    def get_memory(self, job_id: str) -> Optional[MemorySaver]:
        return self._memory_store.get(job_id)

job_manager = SQLiteJobManager()

# ============================================================================
# DATA MODELS
# ============================================================================

class BlogRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200)
    auto_approve: bool = Field(True, description="Skip human plan review")
    include_images: bool = Field(True)
    include_podcast: bool = Field(True)
    tone: Optional[str] = None
    audience: Optional[str] = None
    
    @classmethod
    def _sanitize_topic(cls, v: str) -> str:
        # Strip HTML tags
        v = re_module.sub(r'<[^>]+>', '', v)
        # Collapse whitespace
        v = re_module.sub(r'\s+', ' ', v).strip()
        # Block obvious prompt injection attempts
        injection_patterns = ['ignore previous', 'system prompt', 'you are now', 'disregard']
        lower = v.lower()
        for pattern in injection_patterns:
            if pattern in lower:
                raise ValueError(f"Topic contains blocked pattern: '{pattern}'")
        return v

class ApproveRequest(BaseModel):
    feedback: Optional[str] = Field(
        None,
        description="Optional plain-text revision notes. If provided the plan "
                    "is refined by the LLM before writing begins."
    )

# ============================================================================
# BLOG GENERATION — TWO-PHASE ASYNC
# ============================================================================

def _thread_config(job_id: str) -> dict:
    return {"configurable": {"thread_id": job_id}}

async def _phase1_research_and_plan(job_id: str, topic: str, settings: dict):
    """
    Run the graph up to (and including) the orchestrator interrupt.
    Stores the generated plan in the DB and sets status = 'awaiting_approval'.
    If auto_approve is True, immediately kicks off phase 2.
    """
    try:
        job_manager.start_job(job_id)

        validator = TopicValidator()
        if not validator.validate(topic)["valid"]:
            job_manager.fail_job(job_id, "Topic validation failed")
            return

        job_manager.update_progress(job_id, "router")

        folders = create_blog_structure(topic)
        memory  = job_manager.get_memory(job_id)   # the saved MemorySaver

        from datetime import date
        initial_state = {
            "topic":       topic,
            "as_of":       date.today().isoformat(),
            "sections":    [],
            "blog_folder": folders["base"],
            "_job_id":     job_id,
        }

        app_graph = build_graph(memory=memory)    # pass the SAME memory object
        thread    = _thread_config(job_id)

        # Stream phase 1 — graph will pause after orchestrator
        _emit(job_id, "pipeline", "started", "Blog generation pipeline started")

        for event in app_graph.stream(initial_state, thread, stream_mode="values"):
            if event.get("needs_research") is not None:
                job_manager.update_progress(job_id, "research")
            if event.get("plan"):
                job_manager.update_progress(job_id, "planning")

        # Read the plan from the checkpoint
        current_state = app_graph.get_state(thread).values
        plan = current_state.get("plan")

        if not plan:
            job_manager.fail_job(job_id, "Orchestrator did not produce a plan")
            return

        job_manager.store_plan(job_id, plan.model_dump())

        # Auto-approve skips human review
        if settings.get("auto_approve", True):
            await _phase2_write(job_id, folders, feedback=None)

    except Exception as e:
        import traceback
        job_manager.fail_job(job_id, f"Phase 1 failed: {e}\n{traceback.format_exc()}")


async def _phase2_write(job_id: str, folders: dict = None, feedback: Optional[str] = None):
    """
    Resume the graph from the interrupt and run to completion.
    Optionally applies LLM plan refinement if feedback is provided.
    """
    try:
        memory = job_manager.get_memory(job_id)
        if memory is None:
            # Memory lost (e.g. server restarted after phase 1).
            job_manager.fail_job(
                job_id,
                "Graph checkpoint lost — the server was restarted between plan "
                "approval and writing. Please submit a new generation request."
            )
            return

        app_graph = build_graph(memory=memory)
        thread    = _thread_config(job_id)

        # Apply human feedback before resuming if provided
        if feedback:
            from main import refine_plan_with_llm
            from Graph.state import Plan
            current_plan_dict = job_manager.get_job(job_id)["plan"]
            current_plan = Plan(**current_plan_dict)
            new_plan = refine_plan_with_llm(current_plan, feedback)
            app_graph.update_state(thread, {"plan": new_plan})
            job_manager.store_plan(job_id, new_plan.model_dump())

        job_manager.start_job(job_id)   # re-mark as running

        for event in app_graph.stream(None, thread, stream_mode="values", recursion_limit=100):
            if event.get("final"):
                job_manager.update_progress(job_id, "writing")
            if event.get("fact_check_report"):
                job_manager.update_progress(job_id, "fact_check")
            if event.get("linkedin_post"):
                job_manager.update_progress(job_id, "social_media")
            if event.get("audio_path"):
                job_manager.update_progress(job_id, "podcast")
            if event.get("quality_evaluation"):
                job_manager.update_progress(job_id, "evaluation")

        _emit(job_id, "pipeline", "completed", "Blog generation complete!")

        final_state = app_graph.get_state(thread).values

        # Reconstruct folders path if we're resuming via the approve endpoint
        if folders is None:
            job = job_manager.get_job(job_id)
            topic = job["topic"]
            # blogs directory was created in phase 1; find the most recent match
            candidates = sorted(
                Path("blogs").glob(f"{topic[:20].lower().replace(' ', '_')}*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if candidates:
                base = str(candidates[0])
                folders = {
                    "base":     base,
                    "content":  f"{base}/content",
                    "social":   f"{base}/social_media",
                    "reports":  f"{base}/reports",
                    "assets":   f"{base}/assets/images",
                    "research": f"{base}/research",
                    "audio":    f"{base}/audio",
                    "metadata": f"{base}/metadata",
                }
            else:
                folders = create_blog_structure(job_manager.get_job(job_id)["topic"])

        saved_files = save_blog_content(folders, final_state)
        generate_readme(folders, saved_files, final_state)

        plan = final_state.get("plan")
        result = {
            "job_id":          job_id,
            "status":          "completed",
            "topic":           final_state.get("topic", ""),
            "blog_title":      plan.blog_title if plan else "Generated Blog",
            "word_count":      len(final_state.get("final", "").split()),
            "quality_score":   (final_state.get("quality_evaluation") or {}).get("final_score", 0),
            "fact_check_score": 0,
            "folder_path":     folders["base"],
            "files": {
                "blog":         [os.path.basename(f) for f in [saved_files.get("blog")] if f],
                "assets":       [os.path.basename(f) for f in [
                                   saved_files.get("linkedin"), saved_files.get("facebook"), 
                                   saved_files.get("youtube"), saved_files.get("twitter"), 
                                   saved_files.get("email"), saved_files.get("landing_page")] if f],
                "reports":      [os.path.basename(f) for f in
                                 [saved_files.get("fact_check"), saved_files.get("quality_eval")] if f],
                "audio":        [os.path.basename(f) for f in [saved_files.get("audio")] if f],
            },
            "download_url":  f"/api/download/{job_id}",
            "generated_at":  datetime.now().isoformat(),
            "email_sequence": final_state.get("email_sequence", ""),
            "twitter_thread": final_state.get("twitter_thread", ""),
            "landing_page":   final_state.get("landing_page", ""),
            "linkedin_post":  final_state.get("linkedin_post", ""),
            "facebook_post":  final_state.get("facebook_post", ""),
            "youtube_script": final_state.get("youtube_script", ""),
        }

        import re
        if final_state.get("fact_check_report"):
            m = re.search(r'Score:\s*(\d+)/10', final_state["fact_check_report"])
            if m:
                result["fact_check_score"] = float(m.group(1))

        _create_zip(folders["base"], job_id)
        job_manager.complete_job(job_id, result)

    except Exception as e:
        import traceback
        job_manager.fail_job(job_id, f"Phase 2 failed: {e}\n{traceback.format_exc()}")


def _create_zip(folder_path: str, job_id: str) -> str:
    import zipfile
    zip_path = f"static/downloads/{job_id}.zip"
    os.makedirs("static/downloads", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                fp = os.path.join(root, file)
                zf.write(fp, os.path.relpath(fp, folder_path))
    return zip_path

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Serve the frontend UI."""
    response = FileResponse("frontend/index.html")
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.get("/api")
async def api_root():
    return {
        "service": "AI Content Factory API",
        "version": "2.0.0",
        "auth": "All write endpoints require X-API-Key header",
        "endpoints": {
            "generate":  "POST /api/generate",
            "suggest":   "POST /api/suggest",
            "status":    "GET  /api/status/{job_id}",
            "approve":   "POST /api/jobs/{job_id}/approve",
            "download":  "GET  /api/download/{job_id}",
            "list_jobs": "GET  /api/jobs",
            "health":    "GET  /api/health",
        }
    }


@app.get("/api/suggest")
async def suggest_topics():
    """
    Returns 4 highly engaging, trending blog topic suggestions.
    No auth required — called on frontend load.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel, Field
    
    class TopicSuggestions(BaseModel):
        suggestions: List[str] = Field(description="List of exactly 4 strings")

    try:
        llm_fast = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        editor = llm_fast.with_structured_output(TopicSuggestions)
        
        # We use a slightly higher temperature (0.7) so it gives different ideas each time.
        response = editor.invoke([
            SystemMessage(content=TOPIC_SUGGESTIONS_SYSTEM)
        ])
        
        return {"suggestions": response.suggestions[:4]}
    except Exception as e:
        print(f"Suggestion API failed: {e}")
        # Fallback suggestions if API fails
        return {"suggestions": [
            "10 AI Tools Every Developer Needs in 2026",
            "How to Build a Scalable SaaS Architecture",
            "The Future of Multi-Agent Systems",
            "Mastering Python for Generative AI"
        ]}


@app.get("/api/health")
async def health_check():
    with _db() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        running = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
        waiting = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='awaiting_approval'").fetchone()[0]
    return {
        "status":              "healthy",
        "timestamp":           datetime.now().isoformat(),
        "jobs_total":          total,
        "jobs_running":        running,
        "jobs_awaiting_approval": waiting,
    }


@app.post("/api/generate")
async def generate_blog(
    request: BlogRequest,
    background_tasks: BackgroundTasks,
    _key: str = Depends(require_api_key),
):
    """
    Start blog generation. Returns a job_id immediately.

    - If auto_approve=True (default) the pipeline runs end-to-end unattended.
    - If auto_approve=False the job pauses after planning with
      status='awaiting_approval'. Call POST /api/jobs/{job_id}/approve to resume.
    """
    _check_rate_limit(_key)
    settings = {
        "auto_approve":    request.auto_approve,
        "include_images":  request.include_images,
        "include_podcast": request.include_podcast,
        "tone":            request.tone,
        "audience":        request.audience,
    }
    job_id = job_manager.create_job(request.topic, settings)
    background_tasks.add_task(_phase1_research_and_plan, job_id, request.topic, settings)

    return {
        "job_id":         job_id,
        "status":         "pending",
        "topic":          request.topic,
        "auto_approve":   request.auto_approve,
        "created_at":     datetime.now().isoformat(),
        "estimated_time": 120,
        "message": (
            "Generation started. Poll /api/status/{job_id}. "
            + ("Will run fully automatically." if request.auto_approve
               else "Will pause for your approval after planning — "
                    "watch for status='awaiting_approval'.")
        ),
    }


@app.post("/api/jobs/{job_id}/approve")
async def approve_plan(
    job_id: str,
    body: ApproveRequest,
    background_tasks: BackgroundTasks,
    _key: str = Depends(require_api_key),
):
    """
    Approve the generated plan and begin writing.

    - Call with an empty body to approve as-is.
    - Pass {"feedback": "Add a section on ethics"} to refine the plan first.

    Only valid when job status is 'awaiting_approval'.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "awaiting_approval":
        raise HTTPException(
            status_code=400,
            detail=f"Job is '{job['status']}', not 'awaiting_approval'. "
                   "Only paused jobs can be approved."
        )

    background_tasks.add_task(_phase2_write, job_id, None, body.feedback)

    return {
        "job_id":   job_id,
        "status":   "resuming",
        "feedback": body.feedback or "none — plan approved as-is",
        "message":  "Writing phase has started. Poll /api/status/{job_id}.",
    }


@app.post("/api/jobs/{job_id}/reject")
async def reject_plan(
    job_id: str,
    _key: str = Depends(require_api_key),
):
    """
    Reject and cancel a job that is awaiting approval.
    The job is marked as failed and can be deleted.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "awaiting_approval":
        raise HTTPException(status_code=400, detail="Job is not awaiting approval")

    job_manager.fail_job(job_id, "Plan rejected by user")
    return {"job_id": job_id, "status": "failed", "reason": "Plan rejected by user"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Poll this endpoint to track progress. No auth required (read-only)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id":       job_id,
        "status":       job["status"],
        "topic":        job["topic"],
        "created_at":   job["created_at"],
        "started_at":   job["started_at"],
        "completed_at": job["completed_at"],
        "progress":     job["progress"],
        "error":        job["error"],
    }

    if job["status"] == "awaiting_approval":
        response["plan"] = job["plan"]
        response["next_step"] = (
            "POST /api/jobs/{job_id}/approve  — with optional {\"feedback\": \"...\"}"
        )

    if job["status"] == "completed":
        response["result"] = job["result"]
        # Attach the actual markdown content so the UI can render it
        try:
            folder = job["result"].get("folder_path")
            blog_files = job["result"].get("files", {}).get("blog", [])
            if folder and blog_files:
                blog_path = os.path.join(folder, "content", blog_files[0])
                if os.path.exists(blog_path):
                    with open(blog_path, "r", encoding="utf-8") as f:
                        response["result"]["content"] = f.read()
                else:
                    response["result"]["content"] = f"Error: {blog_files[0]} not found on disk."
            else:
                response["result"]["content"] = "Error: Blog file path not available."
        except Exception as e:
            response["result"]["content"] = f"Error reading blog.md: {e}"

    return response


@app.get("/api/jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status"),
):
    """List recent jobs. No auth required (read-only)."""
    jobs = job_manager.list_jobs(limit=limit)
    if status:
        jobs = [j for j in jobs if j["status"] == status]
    return {
        "count": len(jobs),
        "jobs": [
            {
                "job_id":     j["id"],
                "topic":      j["topic"],
                "status":     j["status"],
                "created_at": j["created_at"],
                "progress":   sum(j["progress"].values()),   # stages completed
            }
            for j in jobs
        ]
    }


@app.get("/api/download/{job_id}")
async def download_blog(job_id: str):
    """Download the completed blog package as a ZIP. No auth required."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job['status']})")

    zip_path = f"static/downloads/{job_id}.zip"
    if not os.path.exists(zip_path) and job["result"]:
        _create_zip(job["result"]["folder_path"], job_id)

    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Zip file not found on disk")

    return FileResponse(path=zip_path, filename=f"blog_{job_id}.zip", media_type="application/zip")


@app.delete("/api/jobs/{job_id}")
async def delete_job(
    job_id: str,
    _key: str = Depends(require_api_key),
):
    """Delete a job record and its associated zip file."""
    if not job_manager.get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    job_manager.delete_job(job_id)

    zip_path = f"static/downloads/{job_id}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)

    return {"message": f"Job {job_id} deleted"}


# ============================================================================
# WEBSOCKET — REAL-TIME AGENT EVENTS
# ============================================================================

@app.websocket("/ws/jobs/{job_id}")
async def websocket_agent_events(websocket: WebSocket, job_id: str):
    """
    Stream real-time agent events for a job.
    Connect to ws://localhost:8000/ws/jobs/{job_id} to receive live updates.
    """
    await websocket.accept()
    queue = subscribe(job_id)
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(event)
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        unsubscribe(job_id, queue)


@app.get("/api/jobs/{job_id}/events")
async def get_job_events(job_id: str):
    """Get all past events for a job (for history replay). No auth required."""
    return {"job_id": job_id, "events": get_history(job_id)}

# ============================================================================
# STARTUP / DIRECTORIES
# ============================================================================

os.makedirs("blogs", exist_ok=True)
os.makedirs("static/downloads", exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Content Factory API v2 ...")
    print("UI:    http://localhost:8000")
    print("Docs:  http://localhost:8000/docs")
    uvicorn.run(app, port=8000)