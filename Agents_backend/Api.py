from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, AsyncGenerator
import json
import asyncio

from Graph.state import AgentState
from main import build_graph

app = FastAPI(
    title="Multi-Agent Blog Generator API",
    description="Runs a LangGraph-powered multi-agent workflow to generate fact-checked blogs",
    version="2.0.0",
)

class BlogRequest(BaseModel):
    topic: str = Field(..., min_length=3, example="Future of AI in Healthcare")

class BlogResponse(BaseModel):
    topic: str
    research_data: Optional[str] = None
    sources: Optional[List[Dict]] = None
    blog_outline: Optional[str] = None
    final_blog_post: Optional[str] = None
    fact_check_report: Optional[str] = None
    error: Optional[str] = None

async def stream_blog_generation(topic: str) -> AsyncGenerator:
    """Stream blog generation progress"""
    try:
        graph = build_graph()
        initial_state: AgentState = {"topic": topic}
        
        # Yield initial status
        yield json.dumps({"status": "starting", "message": "Initializing agents..."}) + "\n"
        
        # Stream from graph with intermediate steps
        for event in graph.stream(initial_state):
            # Extract node name and its output
            for node_name, node_output in event.items():
                if node_name == "researcher":
                    yield json.dumps({
                        "status": "progress",
                        "node": "Researcher",
                        "message": "🔍 Researching topic..."
                    }) + "\n"
                    
                elif node_name == "analyst":
                    yield json.dumps({
                        "status": "progress",
                        "node": "Analyst",
                        "message": "📊 Creating outline..."
                    }) + "\n"
                    
                elif node_name == "writer":
                    yield json.dumps({
                        "status": "progress",
                        "node": "Writer",
                        "message": "✍️ Writing blog post..."
                    }) + "\n"
                    
                elif node_name == "fact_checker":
                    yield json.dumps({
                        "status": "progress",
                        "node": "Fact Checker",
                        "message": "✓ Fact-checking content..."
                    }) + "\n"
        
        # After streaming completes, get final state
        final_state = graph.invoke(initial_state)
        
        yield json.dumps({
            "status": "complete",
            "data": final_state
        }) + "\n"
        
    except Exception as e:
        yield json.dumps({
            "status": "error",
            "message": str(e)
        }) + "\n"

@app.post("/generate-blog")
async def generate_blog(request: BlogRequest):
    topic = request.topic.strip()
    
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    return StreamingResponse(
        stream_blog_generation(topic),
        media_type="application/x-ndjson"
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}