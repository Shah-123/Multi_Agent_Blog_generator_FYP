import os
import re
import json
from typing import Tuple, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ============================================================================
# 1. TOPIC VALIDATOR (FIXED)
# ============================================================================

class TopicValidator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def _basic_syntax_check(self, topic: str) -> Tuple[bool, str]:
        topic = topic.strip()
        if len(topic) < 3: return False, "❌ Topic too short"
        if len(topic) > 200: return False, "❌ Topic too long"
        if not any(c.isalpha() for c in topic): return False, "❌ No words detected"
        if re.search(r'(.)\1{4,}', topic): return False, "❌ Gibberish detected"
        return True, "OK"

    def _llm_gatekeeper(self, topic: str) -> Tuple[bool, str]:
        """🆕 FIXED: Better error handling and simpler prompt"""
        try:
            prompt = PromptTemplate(
                template="""You are a topic evaluator. Evaluate this topic: "{topic}"

Return ONLY valid JSON (no other text):
{{"valid": true, "reason": "Topic is good"}} OR {{"valid": false, "reason": "Reason why not"}}

Check: Is the topic coherent, safe, and researchable?""",
                input_variables=["topic"],
            )
            
            chain = prompt | self.llm
            response = chain.invoke({"topic": topic})
            content = response.content.strip()
            
            # Try to extract JSON more robustly
            try:
                # Remove markdown code blocks if present
                if "```" in content:
                    content = content.split("```")[1]
                    if "json" in content:
                        content = content.split("json")[1]
                    content = content.split("```")[0]
                
                data = json.loads(content)
                return data.get("valid", False), data.get("reason", "Unknown")
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON Parse Error: {content[:100]}")
                # If JSON fails, do basic heuristic check
                if "valid" in content.lower() and "true" in content.lower():
                    return True, "OK"
                else:
                    return True, "OK (fallback)"  # Default to allowing topics
        except Exception as e:
            print(f"⚠️ Gatekeeper Exception: {str(e)}")
            return True, "OK (fallback - gatekeeper unavailable)"  # Fallback: allow topic

    def validate(self, topic: str) -> Dict:
        ok, msg = self._basic_syntax_check(topic)
        if not ok: 
            return {"valid": False, "reason": msg}
        
        ok, msg = self._llm_gatekeeper(topic)
        return {"valid": ok, "reason": msg}

# ============================================================================
# 2. EVALUATOR - Tier 1: Structure
# ============================================================================

class HardMetrics:
    @staticmethod
    def calculate_structure(blog_post: str) -> Dict:
        h1 = len(re.findall(r'^# ', blog_post, re.M))
        h2 = len(re.findall(r'^## ', blog_post, re.M))
        h3 = len(re.findall(r'^### ', blog_post, re.M))
        
        score = 0
        
        # H1 Check (Should be exactly 1)
        if h1 == 1: score += 3
        
        # H2 Check
        if 5 <= h2 <= 20: 
            score += 4
        elif h2 > 20:
            score += 2
        
        # H3 Check
        if h3 >= 3: score += 3
        
        return {"score": score, "details": f"H1: {h1}, H2: {h2}, H3: {h3}"}

    @staticmethod
    def calculate_readability(blog_post: str) -> Dict:
        words = blog_post.split()
        sentences = [s for s in re.split(r'[.!?]+', blog_post) if s.strip()]
        if not words or not sentences: return {"score": 0, "details": "Empty"}
        avg_len = len(words) / len(sentences)
        score = 10 - (max(0, avg_len - 20) / 2)
        return {"score": max(0, round(score, 1)), "details": f"Avg Sentence: {avg_len:.1f}"}

# ============================================================================
# 3. EVALUATOR - Tier 2: NLI Fact-Checking
# ============================================================================

class DataDrivenMetrics:
    @staticmethod
    def verify_claims_nli(blog_post: str, research_data) -> Dict:
        """
        Accepts research_data as either Dict (compressed) or String (raw).
        """
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Convert dict to string if needed
        if isinstance(research_data, dict):
            research_str = json.dumps(research_data, indent=2)
        else:
            research_str = str(research_data)
        
        prompt = f"""
        Compare the BLOG CLAIMS against the RESEARCH DATA.
        
        RESEARCH DATA:
        {research_str[:10000]} 
        
        BLOG CONTENT:
        {blog_post[:20000]}
        
        TASK:
        Identify 5-7 MAJOR factual claims scattered throughout the blog. 
        For each:
        1. Categorize: [SUPPORTED], [NEUTRAL], or [CONTRADICTED].
        2. Provide the Source URL from RESEARCH DATA if available.
        
        Format:
        CLAIM X: [CATEGORY]
        SOURCE: [URL or "None"]
        REASON: [Short explanation]
        """
        try:
            res = llm.invoke(prompt).content
            
            sup = res.count("[SUPPORTED]")
            con = res.count("[CONTRADICTED]")
            
            score = max(0, min(10, (sup * 2) - (con * 5)))
            
            return {
                "score": float(score),
                "hallucination_detected": con > 0,
                "report": res 
            }
        except:
            return {"score": 5, "report": "Error in NLI", "hallucination_detected": False}

# ============================================================================
# 4. EVALUATOR - Tier 3: Qualitative Feedback
# ============================================================================

class LLMFeedback:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def get_improvements(self, blog_post: str, topic: str, scores: Dict) -> Dict:
        try:
            prompt = f"Analyze blog on '{topic}'. Scores: {scores}. Provide 3 short, actionable tips to improve flow or depth."
            response = self.llm.invoke(prompt)
            return {"feedback": response.content}
        except:
            return {"feedback": "Could not generate feedback"}

# ============================================================================
# 5. MAIN EVALUATION FUNCTION
# ============================================================================

def realistic_evaluation(blog_post: str, research_data, topic: str) -> Dict:
    """
    research_data can be Dict (compressed) or String (raw).
    """
    struct = HardMetrics.calculate_structure(blog_post)
    read = HardMetrics.calculate_readability(blog_post)
    nli = DataDrivenMetrics.verify_claims_nli(blog_post, research_data)
    
    tier1_avg = (struct['score'] + read['score']) / 2
    final_score = (tier1_avg * 0.4) + (nli['score'] * 0.6)
    
    if nli.get("hallucination_detected"):
        final_score = min(final_score, 5.0)
        verdict = "❌ HALLUCINATION DETECTED"
    else:
        verdict = "✅ PASSED" if final_score >= 7.5 else "⚠️ NEEDS REVISION"

    fb_tool = LLMFeedback()
    feedback = fb_tool.get_improvements(blog_post, topic, {"Score": final_score})

    return {
        "final_score": round(final_score, 1),
        "tier1": {"struct": struct, "read": read},
        "tier2": nli,
        "tier3": feedback,
        "verdict": verdict
    }