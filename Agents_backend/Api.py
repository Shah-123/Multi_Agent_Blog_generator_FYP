from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from Graph.state import AgentState
from main import build_graph

# ------------------------------------------------------------------
# FastAPI App
# ------------------------------------------------------------------

app = FastAPI(
    title="Multi-Agent Blog Generator API",
    description="Runs a LangGraph-powered multi-agent workflow to generate fact-checked blogs",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Request / Response Models
# ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# API Endpoint
# ------------------------------------------------------------------

@app.post("/generate-blog", response_model=BlogResponse)
def generate_blog(request: BlogRequest):
    topic = request.topic.strip()

    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")

    try:
        graph = build_graph()
        initial_state: AgentState = {"topic": topic}

        final_state = graph.invoke(initial_state)

        if final_state.get("error"):
            raise HTTPException(status_code=500, detail=final_state["error"])

        return final_state

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
