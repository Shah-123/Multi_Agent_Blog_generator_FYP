import uuid
import os
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
from threading import Thread

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Import your backend logic
from main import build_graph
from validators import TopicValidator

# 1. SETUP
load_dotenv()
app = FastAPI(title="AI Content Factory Pro", version="2.0.0")

# CORS (Allow Frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def validate_api_key(key: str = Security(api_key_header)):
    # If no API key set in env, allow access (for testing)
    if not API_KEY:
        return "dev-mode"
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return key

# IN-MEMORY DATABASE (Resets when you restart server)
# In production, replace this with Supabase
jobs_db: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class BlogRequest(BaseModel):
    topic: str
    tone: str = "Professional"
    plan: str = "premium" # basic or premium

class JobResponse(BaseModel):
    job_id: str

# ---------------------------------------------------------------------------
# WORKFLOW RUNNER (The Background Worker)
# ---------------------------------------------------------------------------

def run_workflow_sync(job_id: str, topic: str, tone: str, plan: str):
    """Run the Agent in a background thread."""
    print(f"🚀 [Job {job_id}] Started processing: {topic}")
    
    try:
        # 1. Update Status to 'Processing'
        jobs_db[job_id]["status"] = "PROCESSING"
        jobs_db[job_id]["stage"] = "Agent is working..."
        
        # 2. Initialize State (Must match your AgentState definition)
        initial_state = {
            "topic": topic,
            "tone": tone,
            "plan": plan,
            "iteration_count": 0,
            "error": None,
            "sources": [],
            
            # Data Containers
            "research_data": None,      # Structured object
            "raw_research_data": "",    # String fallback
            "competitor_headers": "",
            "blog_outline": None,
            "sections": [],
            "seo_metadata": {},
            "final_blog_post": "",
            "fact_check_report": None,
            "image_path": ""
        }
        
        # 3. Build Graph
        app_graph = build_graph()
        
        # 4. RUN THE AGENT (Only Once!)
        # We use .invoke() for stability.
        final_output = app_graph.invoke(initial_state)
        
        # 5. Handle Failure inside Graph
        if final_output.get("error"):
            raise Exception(final_output["error"])

        # 6. Extract Results
        # Handle Quality Score (Extract form Pydantic or Dict)
        quality_data = final_output.get("quality_evaluation", {})
        score = quality_data.get("final_score", 0) if isinstance(quality_data, dict) else 0
        
        # Handle SEO
        seo = final_output.get("seo_metadata", {})
        
        # Handle Image
        img = final_output.get("image_path")

        # 7. Save to 'Database'
        jobs_db[job_id].update({
            "status": "COMPLETED",
            "stage": "Finished",
            "content": final_output.get("final_blog_post", ""),
            "quality_score": score,
            "seo_metadata": seo,
            "image_path": img,
            "completed_at": datetime.utcnow().isoformat() + "Z"
        })
        print(f"✅ [Job {job_id}] Finished Successfully")

    except Exception as e:
        print(f"❌ [Job {job_id}] Failed: {e}")
        jobs_db[job_id].update({
            "status": "FAILED",
            "stage": "Error occurred",
            "error_msg": str(e)
        })

# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {"message": "AI Blog Factory API is Running 🚀"}

@app.post("/generate", response_model=JobResponse)
async def start_generation(
    req: BlogRequest, 
    key: str = Security(validate_api_key)
):
    """Endpoint for UI to start a blog generation."""
    
    # 1. Validate Input
    validator = TopicValidator()
    check = validator.validate(req.topic)
    if not check["valid"]:
        raise HTTPException(status_code=400, detail=check["reason"])

    # 2. Create Job ID
    job_id = str(uuid.uuid4())
    
    # 3. Store Initial Record
    jobs_db[job_id] = {
        "id": job_id,
        "topic": req.topic,
        "tone": req.tone,
        "plan": req.plan,
        "status": "PENDING",
        "stage": "Queued",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # 4. Start Background Thread
    # We use Threading because LangGraph is synchronous code running inside Async FastAPI
    thread = Thread(target=run_workflow_sync, args=(job_id, req.topic, req.tone, req.plan))
    thread.start()
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def check_status(job_id: str, key: str = Security(validate_api_key)):
    """Endpoint for UI to poll for results."""
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job