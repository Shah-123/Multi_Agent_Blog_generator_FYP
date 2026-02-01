import re
import json
from typing import Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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
            prompt = f"""Evaluate this blog topic: "{topic}"

            Return ONLY valid JSON with this format:
            {{
                "valid": boolean,
                "reason": "short explanation"
            }}

            Criteria for VALID:
            1. Safe (no hate speech, illegal acts, porn).
            2. Coherent (makes sense grammatically).
            3. Researchable (not a random string of numbers).
            """
            
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean up potential markdown formatting from LLM
            content = content.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(content)
            return data.get("valid", False), data.get("reason", "Unknown reason")
        
        except json.JSONDecodeError:
            # Fallback: if JSON fails, assume it's okay but log it
            print("⚠️ Validator JSON parse failed. Defaulting to True.")
            return True, "Topic accepted (Fallback)"
        
        except Exception as e:
            print(f"⚠️ Gatekeeper Exception: {str(e)}")
            return True, "Topic accepted (System Error)"
    
    def validate(self, topic: str) -> Dict[str, Any]:
        # 1. Cheap check first
        ok, msg = self._basic_syntax_check(topic)
        if not ok: 
            return {"valid": False, "reason": msg}
        
        # 2. Smart check second
        ok, msg = self._llm_gatekeeper(topic)
        return {"valid": ok, "reason": msg}

# ============================================================================
# 2. BLOG EVALUATOR (METRICS + AI CRITIC)
# ============================================================================
class BlogEvaluator:
    """Quality evaluator that combines code-based metrics with AI critique."""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    def evaluate_structure(self, blog_post: str) -> float:
        """Check H1/H2/H3 structure (0-3 points)."""
        h1 = len(re.findall(r'^# ', blog_post, re.M))
        h2 = len(re.findall(r'^## ', blog_post, re.M))
        h3 = len(re.findall(r'^### ', blog_post, re.M))
        
        score = 0
        if h1 >= 1: score += 1      # Has a main title
        if 3 <= h2 <= 20: score += 1 # Reasonable number of sections
        if h3 >= 2: score += 1      # Has subsections (depth)
        
        return score
    
    def evaluate_readability(self, blog_post: str) -> float:
        """Check sentence length and flow (0-2 points)."""
        # Strip code blocks to avoid skewing word counts
        clean_text = re.sub(r'```.*?```', '', blog_post, flags=re.DOTALL)
        
        words = clean_text.split()
        sentences = [s for s in re.split(r'[.!?]+', clean_text) if s.strip()]
        
        if not words or not sentences:
            return 0
        
        avg_len = len(words) / len(sentences)
        
        # Ideal: 10-25 words per sentence for blog posts
        if 10 <= avg_len <= 25:
            return 2
        elif 8 <= avg_len <= 35:
            return 1
        return 0
    
    def evaluate_citations(self, blog_post: str) -> float:
        """Check citation coverage (0-5 points)."""
        # Count Markdown links: [Title](https://...)
        # We look for http to ensure it's a URL
        urls = re.findall(r'\]\(http', blog_post)
        url_count = len(urls)
        
        # Score based on density
        if url_count >= 8:
            return 5
        elif url_count >= 5:
            return 3
        elif url_count >= 2:
            return 1
        return 0
    
    def evaluate(self, blog_post: str, topic: str) -> Dict[str, Any]:
        """Run complete evaluation."""
        
        # Hard metrics (0-10 scale)
        struct_score = self.evaluate_structure(blog_post)
        read_score = self.evaluate_readability(blog_post)
        cite_score = self.evaluate_citations(blog_post)
        
        # Calculate final score
        final_score = struct_score + read_score + cite_score
        # Cap at 10 just in case
        final_score = min(final_score, 10)
        
        # Determine verdict
        if final_score >= 8:
            verdict = "✅ EXCELLENT"
        elif final_score >= 6:
            verdict = "✅ GOOD"
        elif final_score >= 4:
            verdict = "⚠️ NEEDS IMPROVEMENT"
        else:
            verdict = "❌ POOR"
        
        # Get Qualitative Feedback from AI
        try:
            feedback_prompt = f"""Analyze this blog post about '{topic}'. 
            The automated quality score is {final_score}/10.
            
            Provide 2 specific, actionable improvements the writer could make.
            Keep it strictly under 50 words.
            """
            
            feedback = self.llm.invoke(feedback_prompt).content.strip()
        except:
            feedback = "AI Feedback unavailable."
        
        return {
            "final_score": final_score,
            "verdict": verdict,
            "metrics": {
                "structure_score": struct_score,
                "readability_score": read_score,
                "citation_score": cite_score
            },
            "ai_feedback": feedback
        }

# Convenience function if called directly
def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
    evaluator = BlogEvaluator()
    return evaluator.evaluate(blog_post, topic)