from typing import List, Optional
from pydantic import BaseModel, Field

# ============================================================================
# 1. RESEARCHER DATA STRUCTURES
# ============================================================================

class ResearchFinding(BaseModel):
    """A single verified fact with confidence assessment."""
    claim: str = Field(description="The specific fact, statistic, or quote.")
    url: str = Field(description="The source URL supporting this claim.")
    confidence: str = Field(description="HIGH, MEDIUM, LOW, or FLAGGED")
    type: str = Field(description="stat, quote, or fact", default="fact")

class ResearchGap(BaseModel):
    """Missing info that the agent couldn't find."""
    topic: str = Field(description="The topic that is missing or unclear.")
    description: str = Field(description="What specific info is needed.")

class ResearchData(BaseModel):
    """The complete output object from the Researcher Node."""
    executive_summary: str = Field(description="High-level summary of the topic.")
    findings: List[ResearchFinding] = Field(description="List of verified facts and stats.")
    gaps: List[ResearchGap] = Field(description="List of missing information.")
    
    def to_string(self) -> str:
        """Helper to convert object back to a Markdown string for the Writer."""
        output = [f"## Executive Summary\n{self.executive_summary}\n"]
        output.append("## Verified Findings")
        for f in self.findings:
            output.append(f"- [{f.confidence}] {f.claim} ({f.url})")
        return "\n".join(output)

# ============================================================================
# 2. ANALYST DATA STRUCTURES
# ============================================================================

class SEOMetadata(BaseModel):
    title: str = Field(description="SEO optimized title")
    meta_description: str = Field(description="SEO meta description")
    keywords: List[str] = Field(description="Target keywords")

class BlogOutline(BaseModel):
    """The structured output from the Analyst."""
    seo: SEOMetadata = Field(description="SEO strategy data")
    sections: List[str] = Field(description="List of section headers (H2)")
    full_outline: str = Field(description="The full Markdown outline")
    # Note: 'plan' is passed in input, usually not strictly needed in output schema 
    # but helpful if the LLM confirms it.

# ============================================================================
# 3. WRITER DATA STRUCTURES
# ============================================================================

class WrittenSection(BaseModel):
    """A single written section."""
    title: str
    content: str
    citations: List[str] = Field(default_factory=list)

class WrittenBlog(BaseModel):
    """The complete final blog post."""
    sections: List[WrittenSection]
    full_text: str
    
    def to_string(self) -> str:
        return self.full_text

# ============================================================================
# 4. FACT CHECKER DATA STRUCTURES
# ============================================================================

class FactCheckIssue(BaseModel):
    claim: str
    issue_type: str = Field(description="citation_missing, hallucination, or contradiction")
    recommendation: str

class FactCheckReport(BaseModel):
    """The final audit report."""
    score: float = Field(description="0-10 quality score")
    verdict: str = Field(description="READY or NEEDS_REVISION")
    issues: List[FactCheckIssue] = Field(default_factory=list)