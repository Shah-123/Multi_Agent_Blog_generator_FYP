from typing import TypedDict, Optional

class AgentState(TypedDict):
    """
    State dictionary that holds all data shared between agents.
    
    Attributes:
        topic: The research topic provided by the user
        research_data: Structured research findings from the Researcher node
        blog_outline: SEO-optimized blog outline from the Analyst node
        final_blog_post: Complete blog post from the Writer node (future)
        error: Optional error message if something fails
    """
    topic: str
    research_data: str
    blog_outline: str
    final_blog_post: str
    error: Optional[str]