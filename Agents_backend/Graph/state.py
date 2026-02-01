import operator
from typing import Annotated, List, Optional, TypedDict, Union
from pydantic import BaseModel, Field

# ============================================================================
# 1. PYDANTIC MODELS (LLM STRUCTURED OUTPUTS)
# ============================================================================

class RouterDecision(BaseModel):
    """Output schema for the Router Node."""
    needs_research: bool = Field(description="True if the topic requires external search info.")
    mode: str = Field(description="One of: 'closed_book', 'hybrid', 'open_book'")
    reason: str = Field(description="Short explanation of the decision")
    queries: List[str] = Field(description="3-5 search queries if research is needed", default=[])

class EvidenceItem(BaseModel):
    """Schema for a single piece of research evidence."""
    title: str = Field(description="Title of the source")
    url: str = Field(description="URL of the source")
    snippet: str = Field(description="Relevant content excerpt")
    published_at: Optional[str] = Field(description="Date string or None")
    source: str = Field(description="Domain name (e.g. techcrunch.com)")

class EvidencePack(BaseModel):
    """Container for search results."""
    evidence: List[EvidenceItem]

class Task(BaseModel):
    """Schema for a single section writing task."""
    id: int = Field(description="Sequential ID (0, 1, 2...)")
    title: str = Field(description="Section H2 Title")
    goal: str = Field(description="What the reader should learn")
    bullets: List[str] = Field(description="Key points to cover")
    target_words: int = Field(description="Approx word count")
    tags: List[str] = Field(description="SEO tags for this section", default=[])

class Plan(BaseModel):
    """Schema for the entire blog outline."""
    blog_title: str = Field(description="SEO-optimized H1 title")
    tone: str = Field(description="Tone of voice (e.g., 'professional', 'conversational')")
    audience: str = Field(description="Target audience")
    tasks: List[Task] = Field(description="List of sections to write")

class ImageSpec(BaseModel):
    """Schema for a single image generation request."""
    placeholder: str = Field(description="e.g., [[IMAGE_1]]")
    filename: str = Field(description="dashed-slug-filename")
    prompt: str = Field(description="Detailed prompt for the image generator")
    alt: str = Field(description="Alt text for accessibility")
    caption: str = Field(description="Caption to display below the image")

class GlobalImagePlan(BaseModel):
    """Schema for the image placement strategy."""
    md_with_placeholders: str = Field(description="The full blog markdown with [[IMAGE_N]] placeholders inserted")
    images: List[ImageSpec] = Field(description="List of images to generate")

# ============================================================================
# 2. GRAPH STATE (THE MEMORY)
# ============================================================================

class State(TypedDict):
    # INPUTS
    topic: str
    as_of: str  # Date string
    
    # ROUTER OUTPUTS
    needs_research: bool
    mode: str
    queries: List[str]
    recency_days: int
    
    # RESEARCH OUTPUTS
    evidence: List[EvidenceItem]  # Stores clean search results
    
    # PLANNING OUTPUTS
    plan: Plan
    
    # WORKER OUTPUTS (PARALLEL)
    # CRITICAL: operator.add allows multiple workers to append to this list
    # without overwriting each other.
    sections: Annotated[List[tuple], operator.add] 
    
    # REDUCER OUTPUTS
    merged_md: str           # Text combined from sections
    md_with_placeholders: str # Text with [[IMAGE_1]] tags
    image_specs: List[dict]   # Instructions for image generator
    
    # FINAL OUTPUTS
    final: str               # The finished Markdown blog post
    fact_check_report: str   # The audit report text
    
    # SOCIAL MEDIA OUTPUTS
    linkedin_post: str
    youtube_script: str
    facebook_post: str
    
    # EVALUATION
    quality_evaluation: dict