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
    """Quality evaluator that combines code-based metrics with AI critique."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        # FIXED: Configurable weights with sensible defaults
        self.weights = weights or {
            "structure": 0.30,
            "readability": 0.35,
            "citations": 0.25,
            "seo": 0.10  # New: basic SEO check
        }
    
    def evaluate_structure(self, blog_post: str) -> float:
        """Check H1/H2/H3 structure (0-10 points)."""
        # FIXED: More nuanced scoring
        h1 = len(re.findall(r'^# ', blog_post, re.M))
        h2 = len(re.findall(r'^## ', blog_post, re.M))
        h3 = len(re.findall(r'^### ', blog_post, re.M))
        
        score = 0
        
        # Title check
        if h1 == 1:
            score += 3  # Exactly one H1
        elif h1 > 1:
            score += 1  # Multiple H1s is bad
        else:
            score += 0  # No H1 is very bad
        
        # Section structure
        if 3 <= h2 <= 8:  # Ideal range for sections
            score += 4
        elif 1 <= h2 <= 10:  # Acceptable range
            score += 2
        else:
            score += 0  # Too few or too many
        
        # Subsection depth (bonus)
        if h3 >= 2:
            score += 3  # Has subsections = good structure
        elif h3 >= 1:
            score += 1
        
        return min(score, 10)  # Cap at 10
    
    def evaluate_readability(self, blog_post: str) -> float:
        """Check sentence length and flow (0-10 points)."""
        # FIXED: Remove code blocks and extract plain text
        clean_text = re.sub(r'```.*?```', '', blog_post, flags=re.DOTALL)
        clean_text = re.sub(r'\[.*?\]\(.*?\)', '', clean_text)  # Remove links for counting
        
        if not clean_text.strip():
            return 0
        
        # Count sentences (split by .?!)
        sentences = [s.strip() for s in re.split(r'[.!?]+', clean_text) if len(s.strip()) > 10]
        
        if not sentences:
            return 5  # Neutral score for no sentences
        
        # Calculate average sentence length
        total_words = sum(len(s.split()) for s in sentences)
        avg_sentence_len = total_words / len(sentences)
        
        # FIXED: Better scoring based on established readability guidelines
        # Ideal: 15-25 words per sentence for blogs
        if 15 <= avg_sentence_len <= 25:
            return 10  # Perfect
        elif 12 <= avg_sentence_len <= 30:
            return 7   # Good
        elif 8 <= avg_sentence_len <= 35:
            return 5   # Fair
        elif 5 <= avg_sentence_len <= 40:
            return 3   # Poor
        else:
            return 1   # Very poor
    
    def evaluate_citations(self, blog_post: str) -> float:
        """Check citation coverage (0-10 points)."""
        # FIXED: Count proper markdown links with URLs
        # Pattern: [text](http... or https...)
        markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', blog_post)
        
        if not markdown_links:
            return 0
        
        # FIXED: Extract unique domains to avoid counting same source multiple times
        unique_domains = set()
        for text, url in markdown_links:
            # Extract domain from URL
            domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                unique_domains.add(domain_match.group(1).lower())
        
        unique_domain_count = len(unique_domains)
        
        # FIXED: Better scoring based on diversity of sources
        if unique_domain_count >= 5:
            return 10  # Excellent source diversity
        elif unique_domain_count >= 3:
            return 7   # Good diversity
        elif unique_domain_count >= 2:
            return 4   # Some diversity
        elif unique_domain_count == 1:
            return 2   # Only one source
        else:
            return 0
    
    def evaluate_seo(self, blog_post: str, topic: str) -> float:
        """Basic SEO checks (0-10 points)."""
        score = 0
        
        # Check if topic appears in first 100 words
        first_100_words = ' '.join(blog_post.split()[:100]).lower()
        topic_lower = topic.lower()
        
        if topic_lower in first_100_words:
            score += 3
        
        # Check for meta keywords (simple version)
        common_words = first_100_words.split()
        if len(set(common_words)) > 20:  # Decent vocabulary diversity
            score += 3
        
        # Check length
        word_count = len(blog_post.split())
        if 1500 <= word_count <= 3000:  # Ideal blog length
            score += 4
        elif 1000 <= word_count <= 4000:  # Acceptable
            score += 2
        
        return min(score, 10)
    
    def evaluate(self, blog_post: str, topic: str) -> Dict[str, Any]:
        """Run complete evaluation."""
        
        # Calculate individual scores
        struct_raw = self.evaluate_structure(blog_post)
        read_raw = self.evaluate_readability(blog_post)
        cite_raw = self.evaluate_citations(blog_post)
        seo_raw = self.evaluate_seo(blog_post, topic)
        
        # Apply weights
        final_score = (
            struct_raw * self.weights["structure"] +
            read_raw * self.weights["readability"] +
            cite_raw * self.weights["citations"] +
            seo_raw * self.weights["seo"]
        )
        
        # Normalize to 0-10 scale
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
        
        # Get Qualitative Feedback from AI
        try:
            # FIXED: Structured feedback
            from typing import List
            from pydantic import BaseModel, Field
            
            class BlogFeedback(BaseModel):
                strengths: List[str] = Field(description="3 specific strengths")
                improvements: List[str] = Field(description="2 actionable improvements")
                overall_impression: str = Field(description="Brief overall impression")
            
            feedback_model = self.llm.with_structured_output(BlogFeedback)
            
            feedback_prompt = f"""Analyze this blog post about '{topic}'.
            
            BLOG CONTENT:
            {blog_post[:3000]}... [truncated]
            
            Provide specific, actionable feedback."""
            
            feedback_result = feedback_model.invoke([
                SystemMessage(content="You are an experienced blog editor."),
                HumanMessage(content=feedback_prompt)
            ])
            
            ai_feedback = {
                "strengths": feedback_result.strengths,
                "improvements": feedback_result.improvements,
                "overall": feedback_result.overall_impression
            }
        except Exception as e:
            print(f"⚠️ AI Feedback failed: {e}")
            ai_feedback = {
                "error": "AI feedback unavailable",
                "strengths": [],
                "improvements": []
            }
        
        return {
            "final_score": final_score,
            "verdict": verdict,
            "metrics": {
                "structure_score": struct_raw,
                "readability_score": read_raw,
                "citation_score": cite_raw,
                "seo_score": seo_raw,
                "weights_used": self.weights
            },
            "raw_counts": {
                "word_count": len(blog_post.split()),
                "link_count": len(re.findall(r'\[.*?\]\(https?://', blog_post)),
                "unique_domains": len(set(re.findall(r'https?://(?:www\.)?([^/]+)', blog_post)))
            },
            "ai_feedback": ai_feedback
        }

# Convenience function if called directly
def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
    evaluator = BlogEvaluator()
    return evaluator.evaluate(blog_post, topic)

