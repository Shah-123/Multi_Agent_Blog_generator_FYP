from typing import TypedDict, List, Optional, Any

class AgentState(TypedDict):
    topic: str
    research_data: str
    competitor_headers: str
    sources: List[dict]
    sections: List[str] 
    blog_outline: str
    final_blog_post: str
    fact_check_report: str
    quality_evaluation: Optional[Any] # Stores the scores
    iteration_count: int              # Prevents infinite loops
    error: Optional[str]