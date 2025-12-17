from typing import TypedDict, Optional, List

class AgentState(TypedDict):
    """
    State dictionary that holds all data shared between agents.
    
    Attributes:
        topic: The research topic provided by the user
        research_data: Structured research findings from the Researcher node
        sources: List of source URLs and metadata for transparency
        blog_outline: SEO-optimized blog outline from the Analyst node
        final_blog_post: Complete blog post from the Writer node
        fact_check_report: Verification report from the Fact-Checker node
        error: Optional error message if something fails
    """
    topic: str
    research_data: str
    sources: List[dict]
    blog_outline: str
    final_blog_post: str
    fact_check_report: str
    error: Optional[str]