import os
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import your existing blog generator
from main import create_blog_structure, save_blog_content, generate_readme, build_graph
from Graph.state import State
from validators import TopicValidator

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="AI Content Factory API",
    description="Generate blogs, social media content, and podcasts with AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================================
# DATA MODELS
# ============================================================================

class BlogRequest(BaseModel):
    topic: str = Field(..., min_length=3, max_length=200, description="Blog topic")
    auto_approve: bool = Field(True, description="Auto-approve the plan without human review")
    include_images: bool = Field(True, description="Generate images")
    include_podcast: bool = Field(True, description="Generate podcast")
    tone: Optional[str] = Field(None, description="Tone (professional, conversational, etc.)")
    audience: Optional[str] = Field(None, description="Target audience")

class BlogResponse(BaseModel):
    job_id: str
    status: str
    topic: str
    created_at: str
    estimated_time: int
    message: str
    progress: Dict[str, Any] = {}

class BlogResult(BaseModel):
    job_id: str
    status: str
    topic: str
    blog_title: str
    word_count: int
    quality_score: float
    fact_check_score: float
    folder_path: str
    files: Dict[str, List[str]]
    download_url: str
    generated_at: str
    time_taken: float

class BlogPreview(BaseModel):
    title: str
    preview: str
    word_count: int
    sections: int
    generated_at: str
    folder: str

# ============================================================================
# JOB MANAGEMENT
# ============================================================================

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}
    
    def create_job(self, topic: str, settings: Dict) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "id": job_id,
            "topic": topic,
            "status": "pending",
            "progress": {
                "router": False,
                "research": False,
                "planning": False,
                "writing": False,
                "images": False,
                "fact_check": False,
                "social_media": False,
                "podcast": False,
                "evaluation": False
            },
            "settings": settings,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None
        }
        return job_id
    
    def update_progress(self, job_id: str, stage: str):
        if job_id in self.jobs:
            self.jobs[job_id]["progress"][stage] = True
    
    def start_job(self, job_id: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["started_at"] = datetime.now().isoformat()
    
    def complete_job(self, job_id: str, result: Dict):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
            self.results[job_id] = result
    
    def fail_job(self, job_id: str, error: str):
        if job_id in self.jobs:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = error
            self.jobs[job_id]["completed_at"] = datetime.now().isoformat()
    
    def get_job(self, job_id: str) -> Optional[Dict]:
        return self.jobs.get(job_id)
    
    def get_result(self, job_id: str) -> Optional[Dict]:
        return self.results.get(job_id)

job_manager = JobManager()

# ============================================================================
# BLOG GENERATION ENGINE (Async Version)
# ============================================================================

async def generate_blog_async(job_id: str, topic: str, settings: Dict):
    """Async wrapper for blog generation."""
    try:
        job_manager.start_job(job_id)
        
        # 1. Validate topic
        validator = TopicValidator()
        validation_result = validator.validate(topic)
        
        if not validation_result["valid"]:
            job_manager.fail_job(job_id, f"Topic validation failed: {validation_result['reason']}")
            return
        
        job_manager.update_progress(job_id, "router")
        
        # 2. Create folder structure
        folders = create_blog_structure(topic)
        job_manager.update_progress(job_id, "research")
        
        # 3. Initial state
        from datetime import date
        thread_id = job_id.replace("-", "_")
        
        initial_state = {
            "topic": topic,
            "as_of": date.today().isoformat(),
            "sections": [],
            "blog_folder": folders["base"],
            "_job_id": job_id
        }
        
        # 4. Build and run graph
        app_graph = build_graph()
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        # Phase 1: Research & Planning
        for event in app_graph.stream(initial_state, thread_config, stream_mode="values"):
            # Update progress based on node
            if hasattr(event, '__getitem__') and 'node' in event:
                node_name = event['node']
                if node_name in ["router", "research", "orchestrator"]:
                    job_manager.update_progress(job_id, node_name)
        
        job_manager.update_progress(job_id, "planning")
        
        # Get current state for plan approval
        current_state = app_graph.get_state(thread_config).values
        plan = current_state.get("plan")
        
        if not plan:
            job_manager.fail_job(job_id, "Failed to generate plan")
            return
        
        # Auto-approve if configured
        if settings.get("auto_approve", True):
            print(f"   🤖 Auto-approving plan for job {job_id}")
        else:
            # In a real API, you might want to store the plan and wait for approval
            # For now, we auto-approve
            pass
        
        # Phase 2: Writing & Polish
        for event in app_graph.stream(None, thread_config, stream_mode="values", recursion_limit=100):
            if hasattr(event, '__getitem__') and 'node' in event:
                node_name = event['node']
                if node_name in ["worker", "merge_content", "decide_images", "generate_and_place_images"]:
                    job_manager.update_progress(job_id, "writing")
                elif node_name == "fact_checker":
                    job_manager.update_progress(job_id, "fact_check")
                elif node_name == "social_media":
                    job_manager.update_progress(job_id, "social_media")
                elif node_name == "audio_generator":
                    job_manager.update_progress(job_id, "podcast")
                elif node_name == "evaluator":
                    job_manager.update_progress(job_id, "evaluation")
        
        # 5. Get final state and save
        final_state = app_graph.get_state(thread_config).values
        saved_files = save_blog_content(folders, final_state)
        readme_file = generate_readme(folders, saved_files, final_state)
        
        # 6. Prepare result
        result = {
            "job_id": job_id,
            "status": "completed",
            "topic": topic,
            "blog_title": plan.blog_title if plan else "Generated Blog",
            "word_count": len(final_state.get("final", "").split()) if final_state.get("final") else 0,
            "quality_score": final_state.get("quality_evaluation", {}).get("final_score", 0) if final_state.get("quality_evaluation") else 0,
            "fact_check_score": 0,  # Will extract from report
            "folder_path": folders["base"],
            "files": {
                "blog": [os.path.basename(f) for f in [saved_files.get("blog")] if f],
                "social_media": [os.path.basename(f) for f in saved_files.get("social", [])],
                "reports": [os.path.basename(f) for f in [saved_files.get("fact_check"), saved_files.get("quality_eval")] if f],
                "assets": [os.path.basename(f) for f in [saved_files.get("audio")] if f]
            },
            "download_url": f"/api/download/{job_id}",
            "generated_at": datetime.now().isoformat(),
            "time_taken": (datetime.now() - datetime.fromisoformat(job_manager.jobs[job_id]["started_at"])).total_seconds()
        }
        
        # Extract fact check score
        if final_state.get("fact_check_report"):
            import re
            match = re.search(r'Score:\s*(\d+)/10', final_state["fact_check_report"])
            if match:
                result["fact_check_score"] = float(match.group(1))
        
        # Create zip file
        create_zip_archive(folders["base"], job_id)
        
        job_manager.complete_job(job_id, result)
        
    except Exception as e:
        job_manager.fail_job(job_id, f"Blog generation failed: {str(e)}")
        import traceback
        print(f"Error in job {job_id}: {traceback.format_exc()}")

def create_zip_archive(folder_path: str, job_id: str):
    """Create a zip archive of the blog folder."""
    import zipfile
    import shutil
    
    zip_path = f"static/downloads/{job_id}.zip"
    os.makedirs("static/downloads", exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    
    return zip_path

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "service": "AI Content Factory API",
        "version": "1.0.0",
        "endpoints": {
            "generate_blog": "POST /api/generate",
            "check_status": "GET /api/status/{job_id}",
            "download_blog": "GET /api/download/{job_id}",
            "list_blogs": "GET /api/blogs",
            "health": "GET /api/health"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len([j for j in job_manager.jobs.values() if j["status"] == "running"]),
        "total_jobs": len(job_manager.jobs)
    }

@app.post("/api/generate", response_model=BlogResponse)
async def generate_blog(request: BlogRequest, background_tasks: BackgroundTasks):
    """
    Generate a complete blog package.
    
    This endpoint starts the blog generation process asynchronously.
    Returns a job ID that can be used to check status and download results.
    """
    # Validate topic length
    if len(request.topic) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters")
    
    # Create job
    settings = {
        "auto_approve": request.auto_approve,
        "include_images": request.include_images,
        "include_podcast": request.include_podcast,
        "tone": request.tone,
        "audience": request.audience
    }
    
    job_id = job_manager.create_job(request.topic, settings)
    
    # Start background task
    background_tasks.add_task(generate_blog_async, job_id, request.topic, settings)
    
    return {
        "job_id": job_id,
        "status": "pending",
        "topic": request.topic,
        "created_at": datetime.now().isoformat(),
        "estimated_time": 120,  # 2 minutes estimate
        "message": "Blog generation started. Use the job_id to check status.",
        "progress": job_manager.get_job(job_id)["progress"]
    }

@app.get("/api/status/{job_id}", response_model=Dict[str, Any])
async def get_job_status(job_id: str):
    """
    Check the status of a blog generation job.
    
    Returns current progress, status, and result when completed.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "topic": job["topic"],
        "created_at": job["created_at"],
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
        "progress": job["progress"],
        "error": job["error"]
    }
    
    # Add result if completed
    if job["status"] == "completed":
        result = job_manager.get_result(job_id)
        if result:
            response["result"] = result
    
    return response

@app.get("/api/download/{job_id}")
async def download_blog(job_id: str):
    """
    Download the generated blog package as a ZIP file.
    
    Only available when the job is completed.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Blog generation not completed yet")
    
    zip_path = f"static/downloads/{job_id}.zip"
    if not os.path.exists(zip_path):
        # Create zip if it doesn't exist
        result = job_manager.get_result(job_id)
        if result and "folder_path" in result:
            create_zip_archive(result["folder_path"], job_id)
    
    if os.path.exists(zip_path):
        return FileResponse(
            path=zip_path,
            filename=f"blog_{job_id}.zip",
            media_type="application/zip"
        )
    else:
        raise HTTPException(status_code=404, detail="Download file not found")

@app.get("/api/blogs", response_model=List[BlogPreview])
async def list_generated_blogs(limit: int = Query(10, ge=1, le=100)):
    """
    List recently generated blogs.
    
    Returns metadata about generated blogs, sorted by creation date.
    """
    blogs = []
    
    # Scan blogs directory
    blogs_dir = Path("blogs")
    if blogs_dir.exists():
        for blog_folder in sorted(blogs_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            if blog_folder.is_dir():
                # Check for metadata
                metadata_file = blog_folder / "metadata" / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Find blog file
                        content_dir = blog_folder / "content"
                        if content_dir.exists():
                            blog_files = list(content_dir.glob("*.md"))
                            if blog_files:
                                blog_file = blog_files[0]
                                preview = blog_file.read_text(encoding='utf-8')[:200]
                                
                                blogs.append({
                                    "title": metadata.get("topic", "Unknown"),
                                    "preview": preview + "..." if len(preview) >= 200 else preview,
                                    "word_count": metadata.get("word_count", 0),
                                    "sections": 0,  # Could count from content
                                    "generated_at": metadata.get("generated_at", ""),
                                    "folder": str(blog_folder.name)
                                })
                    except Exception:
                        continue
    
    return blogs

@app.get("/api/preview/{folder}")
async def preview_blog(folder: str):
    """
    Preview a generated blog by folder name.
    
    Returns the blog content and metadata.
    """
    blog_path = Path("blogs") / folder
    if not blog_path.exists():
        raise HTTPException(status_code=404, detail="Blog folder not found")
    
    # Find blog file
    content_dir = blog_path / "content"
    if not content_dir.exists():
        raise HTTPException(status_code=404, detail="Blog content not found")
    
    blog_files = list(content_dir.glob("*.md"))
    if not blog_files:
        raise HTTPException(status_code=404, detail="No blog file found")
    
    blog_file = blog_files[0]
    content = blog_file.read_text(encoding='utf-8')
    
    # Load metadata
    metadata = {}
    metadata_file = blog_path / "metadata" / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    # Load quality report
    quality_report = {}
    quality_file = blog_path / "reports" / "quality_evaluation.json"
    if quality_file.exists():
        with open(quality_file, 'r') as f:
            quality_report = json.load(f)
    
    return {
        "folder": folder,
        "title": metadata.get("topic", "Unknown"),
        "content": content,
        "metadata": metadata,
        "quality_report": quality_report,
        "word_count": len(content.split()),
        "generated_at": metadata.get("generated_at", "")
    }

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.
    
    Useful for cleaning up completed or failed jobs.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove from memory
    if job_id in job_manager.jobs:
        del job_manager.jobs[job_id]
    
    if job_id in job_manager.results:
        del job_manager.results[job_id]
    
    # Delete download file
    zip_path = f"static/downloads/{job_id}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return {"message": f"Job {job_id} deleted successfully"}

# # ============================================================================
# # WEB INTERFACE ENDPOINTS (Optional)
# # ============================================================================

# @app.get("/ui/generate")
# async def generation_ui():
#     """Simple HTML interface for blog generation."""
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>AI Content Factory</title>
#         <style>
#             body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
#             .container { background: #f5f5f5; padding: 30px; border-radius: 10px; }
#             input, textarea { width: 100%; padding: 10px; margin: 10px 0; }
#             button { background: #007bff; color: white; border: none; padding: 15px 30px; border-radius: 5px; cursor: pointer; }
#             #status { margin-top: 20px; padding: 15px; border-radius: 5px; }
#             .pending { background: #fff3cd; color: #856404; }
#             .running { background: #d1ecf1; color: #0c5460; }
#             .completed { background: #d4edda; color: #155724; }
#             .failed { background: #f8d7da; color: #721c24; }
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h1>🚀 AI Content Factory</h1>
#             <p>Generate complete blog packages with AI</p>
            
#             <div>
#                 <input type="text" id="topic" placeholder="Enter blog topic..." />
#                 <br>
#                 <label><input type="checkbox" id="auto_approve" checked> Auto-approve plan</label>
#                 <br>
#                 <button onclick="generateBlog()">Generate Blog</button>
#             </div>
            
#             <div id="status"></div>
#             <div id="result"></div>
#         </div>
        
#         <script>
#             async function generateBlog() {
#                 const topic = document.getElementById('topic').value;
#                 const autoApprove = document.getElementById('auto_approve').checked;
                
#                 if (!topic) {
#                     alert('Please enter a topic');
#                     return;
#                 }
                
#                 const statusDiv = document.getElementById('status');
#                 statusDiv.innerHTML = '<div class="pending">Starting blog generation...</div>';
                
#                 try {
#                     const response = await fetch('/api/generate', {
#                         method: 'POST',
#                         headers: { 'Content-Type': 'application/json' },
#                         body: JSON.stringify({ topic, auto_approve: autoApprove })
#                     });
                    
#                     const data = await response.json();
                    
#                     if (response.ok) {
#                         statusDiv.innerHTML = `<div class="running">Blog generation started! Job ID: ${data.job_id}</div>`;
#                         checkStatus(data.job_id);
#                     } else {
#                         statusDiv.innerHTML = `<div class="failed">Error: ${data.detail}</div>`;
#                     }
#                 } catch (error) {
#                     statusDiv.innerHTML = `<div class="failed">Network error: ${error}</div>`;
#                 }
#             }
            
#             async function checkStatus(jobId) {
#                 const response = await fetch(`/api/status/${jobId}`);
#                 const data = await response.json();
                
#                 const statusDiv = document.getElementById('status');
#                 const resultDiv = document.getElementById('result');
                
#                 if (data.status === 'completed') {
#                     statusDiv.innerHTML = `<div class="completed">Blog generated successfully!</div>`;
#                     resultDiv.innerHTML = `
#                         <h3>📊 Results</h3>
#                         <p><strong>Title:</strong> ${data.result.blog_title}</p>
#                         <p><strong>Word Count:</strong> ${data.result.word_count}</p>
#                         <p><strong>Quality Score:</strong> ${data.result.quality_score}/10</p>
#                         <p>
#                             <a href="/api/download/${jobId}" target="_blank">
#                                 <button>📥 Download Blog Package</button>
#                             </a>
#                         </p>
#                     `;
#                 } else if (data.status === 'failed') {
#                     statusDiv.innerHTML = `<div class="failed">Failed: ${data.error}</div>`;
#                 } else {
#                     statusDiv.innerHTML = `<div class="running">Processing... (${Object.values(data.progress).filter(v => v).length}/9 stages)</div>`;
#                     setTimeout(() => checkStatus(jobId), 2000);
#                 }
#             }
#         </script>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

from fastapi.responses import HTMLResponse

# Create necessary directories
os.makedirs("blogs", exist_ok=True)
os.makedirs("static/downloads", exist_ok=True)

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("Starting AI Content Factory API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("Web Interface: http://localhost:8000/ui/generate")
    uvicorn.run(app, host="0.0.0.0", port=8000)