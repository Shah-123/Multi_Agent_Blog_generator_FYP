from typing import TypedDict, List, Optional, Any

class AgentState(TypedDict):
    topic: str
    research_data: str
    sources: List[dict]
    blog_outline: str
    final_blog_post: str
    fact_check_report: str
    quality_evaluation: Optional[Any] # Stores the scores
    iteration_count: int              # Prevents infinite loops
    error: Optional[str]
    image_paths: List[str]  # Store paths to generated images