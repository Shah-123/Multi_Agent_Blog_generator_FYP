from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from dotenv import load_dotenv
load_dotenv()

from main import build_graph, regenerate_blog_with_feedback
from validators import TopicValidator, realistic_evaluation
from Graph.nodes import fact_checker_node

app = FastAPI(
    title="Multi-Agent Blog Generator API",
    description="Runs a LangGraph-powered multi-agent workflow to generate fact-checked blogs",
    version="3.0.0",
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class BlogRequest(BaseModel):
    topic: str = Field(..., min_length=3, example="Future of AI in Healthcare")

class BlogResponse(BaseModel):
    topic: str
    research_data: Optional[str] = None
    sources: Optional[List[Dict]] = None
    blog_outline: Optional[str] = None
    final_blog_post: Optional[str] = None
    improved_blog_post: Optional[str] = None
    fact_check_report: Optional[str] = None
    quality_evaluation: Optional[Dict] = None
    error: Optional[str] = None

class BlogRegenerationRequest(BaseModel):
    topic: str
    blog_outline: str
    research_data: str
    llm_feedback: str
    iteration: int = Field(default=1, ge=1, le=5)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post("/generate-blog", response_model=BlogResponse)
async def generate_blog(request: BlogRequest):
    """
    Generate a blog post with automatic quality evaluation and regeneration.
    """
    topic = request.topic.strip()
    
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
    
    try:
        # Step 1: Validate topic
        topic_validator = TopicValidator()
        validation_result = topic_validator.validate(topic)
        
        if not validation_result["valid"]:
            return BlogResponse(
                topic=topic,
                error=f"Topic validation failed: {validation_result['reason']}"
            )
        
        # Step 2: Build graph and run workflow
        graph = build_graph()
        initial_state = {
            "topic": topic,
            "research_data": "",
            "sources": [],
            "blog_outline": "",
            "final_blog_post": "",
            "fact_check_report": "",
            "error": None,
        }
        
        # Run the complete workflow
        final_state = graph.invoke(initial_state)
        
        # Check for errors
        if final_state.get("error"):
            return BlogResponse(
                topic=topic,
                error=final_state["error"]
            )
        
        # Step 3: Evaluate quality
        blog_post = final_state.get("final_blog_post", "")
        research_data = final_state.get("research_data", "")
        
        quality_details = realistic_evaluation(
            blog_post=blog_post,
            research_data=research_data,
            topic=topic
        )
        
        final_state["quality_evaluation"] = quality_details
        
        # Step 4: Auto-regenerate if feedback available
        llm_feedback_obj = quality_details.get("tier3", {}).get("feedback", "")
        improved_blog = None
        
        if llm_feedback_obj:
            improved_blog = regenerate_blog_with_feedback(
                blog_outline=final_state.get("blog_outline", ""),
                research_data=research_data,
                topic=topic,
                llm_feedback=llm_feedback_obj,
                iteration=1
            )
            
            # Fact-check improved blog
            fact_check_state = {
                "topic": topic,
                "research_data": research_data,
                "final_blog_post": improved_blog,
                "error": None
            }
            
            fact_check_result = fact_checker_node(fact_check_state)
            final_state["fact_check_report"] = fact_check_result.get("fact_check_report", "No issues found")
        
        # Return complete response
        return BlogResponse(
            topic=final_state.get("topic"),
            research_data=final_state.get("research_data"),
            sources=final_state.get("sources"),
            blog_outline=final_state.get("blog_outline"),
            final_blog_post=final_state.get("final_blog_post"),
            improved_blog_post=improved_blog,
            fact_check_report=final_state.get("fact_check_report"),
            quality_evaluation=final_state.get("quality_evaluation")
        )
        
    except Exception as e:
        return BlogResponse(
            topic=topic,
            error=f"API Error: {str(e)}"
        )


@app.post("/regenerate-blog")
async def regenerate_blog_manual(request: BlogRegenerationRequest):
    """
    Manually regenerate a blog post with specific feedback.
    """
    try:
        if not all([request.topic, request.blog_outline, request.research_data, request.llm_feedback]):
            raise HTTPException(status_code=400, detail="Missing required fields")
        
        improved_blog = regenerate_blog_with_feedback(
            blog_outline=request.blog_outline,
            research_data=request.research_data,
            topic=request.topic,
            llm_feedback=request.llm_feedback,
            iteration=request.iteration
        )
        
        return {
            "topic": request.topic,
            "iteration": request.iteration,
            "improved_blog_post": improved_blog,
            "blog_length": len(improved_blog),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate-blog")
async def evaluate_blog_quality(blog_data: Dict):
    """
    Evaluate an existing blog post for quality.
    """
    try:
        blog_post = blog_data.get("blog_post")
        topic = blog_data.get("topic")
        research_data = blog_data.get("research_data", "")
        
        if not blog_post or not topic:
            raise HTTPException(status_code=400, detail="blog_post and topic are required")
        
        quality_details = realistic_evaluation(
            blog_post=blog_post,
            research_data=research_data,
            topic=topic
        )
        
        return {
            "topic": topic,
            "quality_evaluation": quality_details,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "3.0.0",
        "features": [
            "blog-generation",
            "quality-evaluation",
            "auto-regeneration",
            "fact-checking"
        ]
    }


@app.get("/")
async def root():
    """API Information"""
    return {
        "name": "Multi-Agent Blog Generator API",
        "version": "3.0.0",
        "endpoints": {
            "POST /generate-blog": "Generate blog with auto-evaluation and regeneration",
            "POST /regenerate-blog": "Manually regenerate blog with feedback",
            "POST /evaluate-blog": "Evaluate existing blog quality",
            "GET /health": "Health check",
            "GET /docs": "API documentation (Swagger UI)"
        }
    }