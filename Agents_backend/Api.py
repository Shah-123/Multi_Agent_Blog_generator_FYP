import uuid
import os
from datetime import datetime
from typing import Dict, Any
from threading import Thread

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

from main import build_graph
from validators import TopicValidator

# ============================================================================
# SETUP
# ============================================================================
load_dotenv()
app = FastAPI(title="AI Content Factory Pro", version="2.1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security (FIXED)
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    print("⚠️ WARNING: No API_KEY set! API is unsecured!")

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def validate_api_key(key: str = Security(api_key_header)):
    """Secure API key validation."""
    if not API_KEY:
        # In production, this should REJECT requests
        print("⚠️ Dev mode: No API key required")
        return "dev-mode"
    
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return key

# In-memory job database
jobs_db: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# DATA MODELS
# ============================================================================
class BlogRequest(BaseModel):
    topic: str
    tone: str = "Professional"
    plan: str = "premium"

class JobResponse(BaseModel):
    job_id: str

# ============================================================================
# BACKGROUND WORKER
# ============================================================================
def run_workflow_sync(job_id: str, topic: str, tone: str, plan: str):
    """Run the agent workflow in background."""
    print(f"🚀 [Job {job_id}] Started: {topic}")
    
    try:
        # Update status
        jobs_db[job_id]["status"] = "PROCESSING"
        jobs_db[job_id]["stage"] = "Agent is working..."
        
        # Initialize state (SIMPLIFIED)
        initial_state = {
            "topic": topic,
            "tone": tone,
            "plan": plan,
            "iteration_count": 0,
            "error": None,
            
            # Data containers
            "citation_index": "",
            "competitor_headers": "",
            "blog_outline": "",
            "sections": [],
            "seo_metadata": {},
            "final_blog_post": "",
            "fact_check_report": "",
            "image_path": "",
            
            # Social media
            "linkedin_post": "",
            "youtube_script": "",
            "facebook_post": "",
            
            # Control
            "quality_evaluation": None
        }
        
        # Build and run graph
        app_graph = build_graph()
        final_output = app_graph.invoke(initial_state)
        
        # Handle failure
        if final_output.get("error"):
            raise Exception(final_output["error"])
        
        # Extract results
        quality_data = final_output.get("quality_evaluation", {})
        score = quality_data.get("final_score", 0)
        
        # Save to database
        jobs_db[job_id].update({
            "status": "COMPLETED",
            "stage": "Finished",
            "content": final_output.get("final_blog_post", ""),
            "quality_score": score,
            "seo_metadata": final_output.get("seo_metadata", {}),
            "image_path": final_output.get("image_path", ""),
            "social_media": {
                "linkedin": final_output.get("linkedin_post", ""),
                "youtube": final_output.get("youtube_script", ""),
                "facebook": final_output.get("facebook_post", "")
            },
            "completed_at": datetime.utcnow().isoformat() + "Z"
        })
        
        print(f"✅ [Job {job_id}] Completed")
    
    except Exception as e:
        print(f"❌ [Job {job_id}] Failed: {e}")
        jobs_db[job_id].update({
            "status": "FAILED",
            "stage": "Error occurred",
            "error_msg": str(e)
        })

# ============================================================================
# API ENDPOINTS
# ============================================================================
@app.get("/")
async def root():
    return {"message": "AI Content Factory API is Running 🚀", "version": "2.1.0"}

@app.post("/generate", response_model=JobResponse)
async def start_generation(
    req: BlogRequest, 
    key: str = Security(validate_api_key)
):
    """Start blog generation."""
    
    # Validate topic
    validator = TopicValidator()
    check = validator.validate(req.topic)
    if not check["valid"]:
        raise HTTPException(status_code=400, detail=check["reason"])
    
    # Create job
    job_id = str(uuid.uuid4())
    jobs_db[job_id] = {
        "id": job_id,
        "topic": req.topic,
        "tone": req.tone,
        "plan": req.plan,
        "status": "PENDING",
        "stage": "Queued",
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    # Start background thread
    thread = Thread(
        target=run_workflow_sync, 
        args=(job_id, req.topic, req.tone, req.plan),
        daemon=True
    )
    thread.start()
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def check_status(job_id: str, key: str = Security(validate_api_key)):
    """Check job status."""
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@app.get("/health")
async def health_check():
    """Health check endpoint (no auth required)."""
    return {"status": "healthy", "jobs_count": len(jobs_db)}

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "Api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )