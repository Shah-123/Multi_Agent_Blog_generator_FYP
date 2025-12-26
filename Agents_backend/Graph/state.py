from typing import TypedDict, List, Optional, Any

class AgentState(TypedDict):
    topic: str
    tone: str
    # Plan specification
    plan: str 
    # Existing fields (keep for backward compatibility)
    research_data: str
    raw_research_data: str
    competitor_headers: str
    sources: List[dict]
    sections: List[str] 
    blog_outline: str
    # 🆕 NEW: Compressed research (Token leak fix)
    compressed_research: dict
    
    # 🆕 NEW: Citation index for writer (Token leak fix)
    citation_index: str
    
    # SEO metadata
    seo_metadata: dict 
    image_path: str
    
    final_blog_post: str
    fact_check_report: str
    quality_evaluation: Optional[Any] 
    iteration_count: int              
    error: Optional[str]