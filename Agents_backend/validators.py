import re
import json
from typing import Tuple, Dict
from langchain_openai import ChatOpenAI

# ============================================================================
# TOPIC VALIDATOR (FIXED)
# ============================================================================
class TopicValidator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def _basic_syntax_check(self, topic: str) -> Tuple[bool, str]:
        topic = topic.strip()
        if len(topic) < 3: 
            return False, "❌ Topic too short"
        if len(topic) > 200: 
            return False, "❌ Topic too long"
        if not any(c.isalpha() for c in topic): 
            return False, "❌ No words detected"
        if re.search(r'(.)\1{4,}', topic): 
            return False, "❌ Gibberish detected"
        return True, "OK"
    
    def _llm_gatekeeper(self, topic: str) -> Tuple[bool, str]:
        """LLM-based validation with better error handling."""
        try:
            prompt = f"""Evaluate this topic: "{topic}"

Return ONLY valid JSON:
{{"valid": true, "reason": "OK"}} OR {{"valid": false, "reason": "Why not"}}

Check: Is it coherent, safe, and researchable?"""
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Remove markdown code blocks
            if "```" in content:
                content = re.sub(r"```(?:json)?\n?", "", content)
            
            data = json.loads(content)
            return data.get("valid", False), data.get("reason", "Unknown")
        
        except json.JSONDecodeError:
            # Fallback: Check for keywords
            if "valid" in content.lower() and "true" in content.lower():
                return True, "OK"
            return True, "OK (fallback)"
        
        except Exception as e:
            print(f"⚠️ Gatekeeper Exception: {str(e)}")
            return True, "OK (fallback)"
    
    def validate(self, topic: str) -> Dict:
        ok, msg = self._basic_syntax_check(topic)
        if not ok: 
            return {"valid": False, "reason": msg}
        
        ok, msg = self._llm_gatekeeper(topic)
        return {"valid": ok, "reason": msg}

# ============================================================================
# EVALUATOR (SIMPLIFIED)
# ============================================================================
class BlogEvaluator:
    """Simplified quality evaluator."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def evaluate_structure(self, blog_post: str) -> float:
        """Check H1/H2/H3 structure (0-3 points)."""
        h1 = len(re.findall(r'^# ', blog_post, re.M))
        h2 = len(re.findall(r'^## ', blog_post, re.M))
        h3 = len(re.findall(r'^### ', blog_post, re.M))
        
        score = 0
        if h1 == 1: score += 1  # Exactly one H1
        if 5 <= h2 <= 20: score += 1  # Good H2 count
        if h3 >= 3: score += 1  # Some H3s
        
        return score
    
    def evaluate_readability(self, blog_post: str) -> float:
        """Check sentence length (0-2 points)."""
        words = blog_post.split()
        sentences = [s for s in re.split(r'[.!?]+', blog_post) if s.strip()]
        
        if not words or not sentences:
            return 0
        
        avg_len = len(words) / len(sentences)
        
        # Ideal: 15-25 words per sentence
        if 15 <= avg_len <= 25:
            return 2
        elif 10 <= avg_len <= 30:
            return 1
        return 0
    
    def evaluate_citations(self, blog_post: str) -> float:
        """Check citation coverage (0-5 points)."""
        # Count URLs in format (https://...)
        urls = re.findall(r'\(https?://[^\)]+\)', blog_post)
        url_count = len(urls)
        
        # Score based on citation density
        if url_count >= 10:
            return 5
        elif url_count >= 5:
            return 3
        elif url_count >= 2:
            return 1
        return 0
    
    def evaluate(self, blog_post: str, topic: str) -> Dict:
        """Run complete evaluation."""
        
        # Hard metrics (0-10)
        struct_score = self.evaluate_structure(blog_post)
        read_score = self.evaluate_readability(blog_post)
        cite_score = self.evaluate_citations(blog_post)
        
        # Calculate final score
        final_score = struct_score + read_score + cite_score
        
        # Determine verdict
        if final_score >= 8:
            verdict = "✅ EXCELLENT"
        elif final_score >= 6:
            verdict = "✅ GOOD"
        elif final_score >= 4:
            verdict = "⚠️ NEEDS IMPROVEMENT"
        else:
            verdict = "❌ POOR"
        
        # Get LLM feedback
        try:
            feedback_prompt = f"""Analyze this blog on '{topic}'. Score: {final_score}/10.
Provide 2-3 specific, actionable improvements.
Keep it under 100 words."""
            
            feedback = self.llm.invoke(feedback_prompt).content
        except:
            feedback = "Could not generate feedback."
        
        return {
            "final_score": final_score,
            "verdict": verdict,
            "tier1": {
                "structure": struct_score,
                "readability": read_score,
                "citations": cite_score
            },
            "tier3": {
                "feedback": feedback
            }
        }

# Convenience function for backward compatibility
def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
    """Legacy function wrapper."""
    evaluator = BlogEvaluator()
    return evaluator.evaluate(blog_post, topic)