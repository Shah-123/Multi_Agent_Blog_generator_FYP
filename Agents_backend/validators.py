import re
import json
from typing import Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment")

# Then pass it explicitly:
# ============================================================================
# 1. TOPIC VALIDATOR (AI-POWERED)
# ============================================================================
class TopicValidator:
    def __init__(self):
        # We use a cheap model for validation to save cost/time
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def _basic_syntax_check(self, topic: str) -> Tuple[bool, str]:
        """Fast, free checks before calling the LLM."""
        topic = topic.strip()
        if len(topic) < 3: 
            return False, "❌ Topic too short"
        if len(topic) > 200: 
            return False, "❌ Topic too long"
        # Check if it has at least one alphabet character
        if not any(c.isalpha() for c in topic): 
            return False, "❌ No words detected"
        # Check for repeated gibberish (e.g. "aaaaaa")
        if re.search(r'(.)\1{4,}', topic): 
            return False, "❌ Gibberish detected"
        return True, "OK"
    
    def _llm_gatekeeper(self, topic: str) -> Tuple[bool, str]:
        """LLM-based validation to check for safety and coherence."""
        try:
            # FIXED: Use structured output for reliable parsing
            from pydantic import BaseModel
            
            class TopicValidation(BaseModel):
                valid: bool
                reason: str
                category: str  # e.g., "safe", "unsafe", "vague", "good"
            
            validator = self.llm.with_structured_output(TopicValidation)
            
            # FIXED: Better prompt with examples
            system_message = """You are a content safety validator. Evaluate if a blog topic is:
1. SAFE: No hate speech, illegal content, porn, or harmful instructions
2. COHERENT: Makes grammatical sense, not random words
3. RESEARCHABLE: Could be researched online, not personal journal entries
4. APPROPRIATE: Suitable for public blog content

Return a structured validation result."""
            
            result = validator.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=f"Evaluate this blog topic: {topic}")
            ])
            
            return result.valid, f"{result.reason} [{result.category}]"
        
        except Exception as e:
            print(f"⚠️ Gatekeeper Exception: {str(e)}")
            return True, "Topic accepted (System Error)"
    
    def validate(self, topic: str) -> Dict[str, Any]:
        """Validate a topic with multiple checks."""
        # 1. Cheap check first
        ok, msg = self._basic_syntax_check(topic)
        if not ok: 
            return {"valid": False, "reason": msg}
        
        # 2. Smart check second
        ok, msg = self._llm_gatekeeper(topic)
        return {"valid": ok, "reason": msg, "topic": topic}

# ============================================================================
# 2. BLOG EVALUATOR (METRICS + AI CRITIC)
# ============================================================================
class BlogEvaluator:
    """Quality evaluator that uses LLM-as-a-Judge to comprehensively grade the content."""
    
    def __init__(self):
        # We use a solid model that can follow complex grading rubrics
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    
    def evaluate(self, blog_post: str, topic: str) -> Dict[str, Any]:
        """Run complete AI evaluation."""
        from typing import List
        from pydantic import BaseModel, Field
        
        # Define the exact grading rubric we want the AI to return
        class BlogFeedback(BaseModel):
            depth_score: float = Field(description="Score 0-10 on how comprehensively and accurately the topic is covered. Penalize fluff or hallucinations.")
            structure_score: float = Field(description="Score 0-10 on logical flow, use of headers (H1, H2, H3), bullet points, and paragraph sizing.")
            readability_score: float = Field(description="Score 0-10 on human-like tone, sentence variety, and avoidance of AI cliches (e.g. 'In conclusion').")
            seo_and_links_score: float = Field(description="Score 0-10 on the presence of inline citations [Source](url) and keyword inclusion.")
            strengths: List[str] = Field(description="3 specific strengths of this article.")
            improvements: List[str] = Field(description="2 actionable improvements.")
            overall_impression: str = Field(description="Brief overall impression.")

        # Bind the schema to the LLM
        evaluator = self.llm.with_structured_output(BlogFeedback)
        
        system_message = """You are an elite blog editor and content quality rater.
Your job is to read the provided blog post (in Markdown) and grade it strictly against four criteria.
DO NOT GIVE PERFECT SCORES easily. Be highly critical.
- Depth: Is it actually valuable, or just generic fluff?
- Structure: Does it look like a well-formatted web article?
- Readability: Does it sound like a human wrote it? Penalize robotic transitions.
- SEO & Links: Does it cite its sources well using [Link](url)?
Return the scores as exact floats (e.g., 7.5, 8.0, 9.2)."""

        # Provide a truncated but large enough chunk for the AI to read
        # GPT-4o-mini can handle 128k tokens, so we can send the whole thing easily,
        # but truncating to 15000 chars is safe to save costs if the blog is huge.
        human_message = f"TOPIC: {topic}\n\nBLOG CONTENT:\n{blog_post[:15000]}"
        
        try:
            result: BlogFeedback = evaluator.invoke([
                SystemMessage(content=system_message),
                HumanMessage(content=human_message)
            ])
            
            # Calculate final score equally weighting the 4 criteria
            final_score = (result.depth_score + result.structure_score + result.readability_score + result.seo_and_links_score) / 4.0
            final_score = round(final_score, 1)
            
            # Determine verdict
            if final_score >= 8.5:
                verdict = "✅ EXCELLENT"
            elif final_score >= 7.0:
                verdict = "✅ GOOD"
            elif final_score >= 5.5:
                verdict = "⚠️ NEEDS IMPROVEMENT"
            else:
                verdict = "❌ POOR"
                
            ai_feedback = {
                "strengths": result.strengths,
                "improvements": result.improvements,
                "overall": result.overall_impression
            }
            
            metrics = {
                "depth_score": result.depth_score,
                "structure_score": result.structure_score,
                "readability_score": result.readability_score,
                "seo_score": result.seo_and_links_score
            }
            
        except Exception as e:
            print(f"⚠️ AI Feedback failed: {e}")
            final_score = 0.0
            verdict = "❌ EVALUATION_FAILED"
            metrics = {}
            ai_feedback = {
                "error": str(e),
                "strengths": [],
                "improvements": []
            }
            
        # Keep raw counts just for informational purposes
        raw_counts = {
            "word_count": len(blog_post.split()),
            "link_count": len(re.findall(r'\[.*?\]\(https?://', blog_post)),
            "unique_domains": len(set(re.findall(r'https?://(?:www\.)?([^/]+)', blog_post)))
        }
        
        return {
            "final_score": final_score,
            "verdict": verdict,
            "metrics": metrics,
            "raw_counts": raw_counts,
            "ai_feedback": ai_feedback
        }

# Convenience function if called directly
def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
    evaluator = BlogEvaluator()
    return evaluator.evaluate(blog_post, topic)

