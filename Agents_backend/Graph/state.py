from typing import TypedDict, List, Optional, Any
from Graph.structured_data import ResearchData, BlogOutline, WrittenBlog, FactCheckReport

class AgentState(TypedDict):
    # ... existing fields ...
    topic: str
    tone: str
    plan: str
    
    # ... existing structured data ...
    research_data: Optional[ResearchData]
    blog_outline: Optional[BlogOutline]
    written_blog: Optional[WrittenBlog]
    fact_check_report: Optional[FactCheckReport]
    
    # ... existing raw data ...
    raw_research_data: str 
    competitor_headers: str
    final_blog_post: str
    seo_metadata: dict
    image_path: str
    
    # 🆕 NEW SOCIAL MEDIA FIELDS
    linkedin_post: str
    youtube_script: str
    facebook_post: str
    
    # ... existing control fields ...
    iteration_count: int
    error: Optional[str]
    sections: List[str]
    sources: List[dict]
    quality_evaluation: Optional[Any]