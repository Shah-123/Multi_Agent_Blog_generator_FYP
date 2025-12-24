import json
import uuid
import asyncio
from fastapi import FastAPI, HTTPException, Security, Depends, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

# --- INTERNAL IMPORTS ---
from main import build_graph, TopicValidator
from database import SessionLocal, BlogRecord, init_db
from dotenv import load_dotenv
import os 
load_dotenv()

# Initialize database on startup
init_db()

app = FastAPI(
    title="AI Content Factory Pro",
    description="Enterprise Multi-Agent Blog Generation with Background Task Support.",
    version="2.0.0"
)




# ADD THIS IMMEDIATELY AFTER app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All your @app.get() and @app.post() routes go AFTER this

# ---------------------------------------------------------------------------
# SECURITY & DB DEPENDENCIES
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-KEY")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def validate_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized API Key")
    return key

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------
class BlogRequest(BaseModel):
    topic: str = Field(..., example="The impact of AI on software engineering")

# ---------------------------------------------------------------------------
# BACKGROUND WORKER (The "Task Engine")
# ---------------------------------------------------------------------------
def run_blog_workflow(blog_id: str, topic: str):
    """Heavy AI logic running in the background."""
    # We create a new DB session for the background thread
    db = SessionLocal()
    try:
        # 1. Run the Graph
        graph = build_graph()
        final_state = graph.invoke({
            "topic": topic,
            "iteration_count": 0,
            "sources": [],
            "research_data": "",
            "blog_outline": "",
            "final_blog_post": "",
            "fact_check_report": ""
        })

        # 2. Extract results
        eval_data = final_state.get("quality_evaluation", {})
        
        # 3. Update Database
        blog = db.query(BlogRecord).filter(BlogRecord.id == blog_id).first()
        if blog:
            blog.content = final_state.get("final_blog_post")
            blog.score = eval_data.get("final_score")
            blog.verdict = eval_data.get("verdict")
            blog.fact_check = str(eval_data.get("tier2", {}).get("report", ""))
            blog.status = "COMPLETED"
            db.commit()
            print(f"✅ Blog {blog_id} completed successfully.")

    except Exception as e:
        print(f"❌ Background Task Error: {str(e)}")
        blog = db.query(BlogRecord).filter(BlogRecord.id == blog_id).first()
        if blog:
            blog.status = f"FAILED: {str(e)}"
            db.commit()
    finally:
        db.close()

# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/generate-async", tags=["Generation"])
async def generate_async_blog(
    request: BlogRequest, 
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    api_key: str = Depends(validate_api_key)
):
    """
    Starts a generation task and returns a Job ID immediately.
    No waiting for the AI.
    """
    # 1. Pre-Gatekeeper check (fast)
    validator = TopicValidator()
    validation = validator.validate(request.topic)
    if not validation["valid"]:
        raise HTTPException(status_code=400, detail=validation["reason"])

    # 2. Create Job ID and Pending Record
    blog_id = str(uuid.uuid4())
    new_blog = BlogRecord(id=blog_id, topic=request.topic, status="PENDING")
    db.add(new_blog)
    db.commit()

    # 3. Queue the background task
    background_tasks.add_task(run_blog_workflow, blog_id, request.topic)

    return {
        "status": "Task Started",
        "job_id": blog_id,
        "check_status_at": f"/status/{blog_id}"
    }

@app.get("/status/{job_id}", tags=["Monitoring"])
def get_job_status(job_id: str, db: Session = Depends(get_db), api_key: str = Depends(validate_api_key)):
    """Check the status and get the result of a specific job."""
    blog = db.query(BlogRecord).filter(BlogRecord.id == job_id).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    return {
        "id": blog.id,
        "topic": blog.topic,
        "status": blog.status,
        "quality_score": blog.score,
        "created_at": blog.created_at,
        "content": blog.content if blog.status == "COMPLETED" else None
    }

@app.get("/history", tags=["Management"])
def get_history(db: Session = Depends(get_db), api_key: str = Depends(validate_api_key)):
    """List all previous generations."""
    return db.query(BlogRecord).order_by(BlogRecord.created_at.desc()).all()