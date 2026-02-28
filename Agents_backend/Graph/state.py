import operator
from typing import Annotated, List, Optional, TypedDict, Any
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
    authors: Optional[str] = Field(description="Author Names or Organization", default=None)

class EvidencePack(BaseModel):
    """Container for search results."""
    evidence: List[EvidenceItem]

class Task(BaseModel):
    """Schema for a single section writing task."""
    id: int = Field(description="Sequential ID (0, 1, 2...)")
    title: str = Field(description="Section H2 Title")
    goal: str = Field(description="What the reader should learn")
    bullets: List[str] = Field(description="Key points to cover")
    target_words: int = Field(default=350, description="Approx word count (default 350)")
    tags: List[str] = Field(description="SEO tags for this section", default=[])

class Plan(BaseModel):
    """Schema for the entire blog outline."""
    blog_title: str = Field(description="SEO-optimized H1 title")
    tone: str = Field(description="Tone of voice (e.g., 'professional', 'conversational')")
    audience: str = Field(description="Target audience")
    tasks: List[Task] = Field(description="List of sections to write")
    
    # NEW: Keyword optimization fields
    primary_keywords: List[str] = Field(
        description="Main SEO keywords to optimize for",
        default=[]
    )
    keyword_strategy: str = Field(
        description="How keywords will be distributed across sections",
        default=""
    )

class ImageSpec(BaseModel):
    """Schema for a single image generation request."""
    target_paragraph: str = Field(description="The exact first 5 words of the paragraph after which this image should be inserted")
    filename: str = Field(description="dashed-slug-filename")
    prompt: str = Field(description="Detailed prompt for the image generator")
    alt: str = Field(description="Alt text for accessibility")
    caption: str = Field(description="Caption to display below the image")

class GlobalImagePlan(BaseModel):
    """Schema for the image placement strategy."""
    images: List[ImageSpec] = Field(description="List of images to generate")

# ============================================================================
# 2. GRAPH STATE (THE MEMORY)
# ============================================================================

class State(TypedDict, total=False):
    """
    The central memory state of the graph.
    'total=False' allows keys to be missing during initialization.
    """
    
    # --- Internal ---
    _job_id: str        # For real-time event emission

    # --- Inputs ---
    topic: str
    as_of: str          # Date string
    blog_folder: str    # Path to save outputs
    # NEW: Tone and keyword inputs
    target_tone: Optional[str]        # e.g., "professional", "conversational"
    target_keywords: List[str]        # e.g., ["AI healthcare", "medical automation"]

    # --- Router Outputs ---
    needs_research: bool
    mode: str
    queries: List[str]
    recency_days: int

    # --- Research Outputs ---
    evidence: List[EvidenceItem]

    # --- Planning Outputs ---
    plan: Plan

    # --- Worker Outputs (Parallel) ---
    # CRITICAL: Annotated[..., operator.add] enables the "Fan-Out" pattern.
    # It tells LangGraph: "When multiple nodes return 'sections', append them 
    # to this list instead of overwriting."
    sections: Annotated[List[tuple], operator.add]

    # --- Reducer/Merger Outputs ---
    merged_md: str            # Text combined from sections
    md_with_placeholders: str # Text with [[IMAGE_1]] tags
    
    # Note: We store dicts here because we use .model_dump() in nodes.py
    image_specs: List[dict]   

    # --- Final Outputs ---
    final: str                # The finished Markdown blog post
    fact_check_report: str    # The audit report text
    
    # --- Self-Healing Fact-Check Loop ---
    fact_check_verdict: str          # "READY" or "NEEDS_REVISION"
    fact_check_issues: List[dict]    # Structured list of flagged issues
    fact_check_score: int            # 0-10 score from fact-checker
    fact_check_attempts: int         # Retry counter (capped at 2)

    # --- Campaign Outputs ---
    linkedin_post: str
    youtube_script: str
    facebook_post: str
    email_sequence: str
    twitter_thread: str
    landing_page: str

    # --- Podcast Outputs ---
    audio_path: Optional[str]   # Path to the MP3 file
    script_path: Optional[str]  # Path to the text script

    # --- Evaluation ---
    completion_report: str          # Text report from validator
    completion_score: int           # 0-10 completeness score
    completion_issues: List[dict]   # Structured list of issues
    quality_evaluation: dict
    
    # NEW: Keyword optimization outputs
    keyword_analysis: dict      # Detailed keyword metrics
    keyword_report: str         # Human-readable report