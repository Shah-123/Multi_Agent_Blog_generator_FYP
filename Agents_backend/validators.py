import os
import re
import json
from typing import Tuple, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ============================================================================
# 1. TOPIC VALIDATOR
# ============================================================================

class TopicValidator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)

    def _basic_syntax_check(self, topic: str) -> Tuple[bool, str]:
        topic = topic.strip()
        if len(topic) < 3: return False, "❌ Topic too short"
        if len(topic) > 200: return False, "❌ Topic too long"
        if not any(c.isalpha() for c in topic): return False, "❌ No words detected"
        if re.search(r'(.)\1{4,}', topic): return False, "❌ Gibberish detected"
        return True, "OK"

    def _llm_gatekeeper(self, topic: str) -> Tuple[bool, str]:
        prompt = PromptTemplate(
            template="""Evaluate the blog topic: "{topic}"
            Return ONLY a JSON object: {{"valid": bool, "reason": "str"}}
            Criteria: Coherence, Safety, Researchability.""",
            input_variables=["topic"],
        )
        try:
            chain = prompt | self.llm
            res = chain.invoke({"topic": topic})
            data = json.loads(res.content.replace("```json", "").replace("```", "").strip())
            return data.get("valid", False), data.get("reason", "Error")
        except:
            return False, "Gatekeeper Error"

    def validate(self, topic: str) -> Dict:
        ok, msg = self._basic_syntax_check(topic)
        if not ok: return {"valid": False, "reason": msg}
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
        if h1 == 1: score += 3
        if 3 <= h2 <= 7: score += 4
        if h3 >= 2: score += 3
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
    def verify_claims_nli(blog_post: str, research_data: str) -> Dict:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # We pass the research data and ask the LLM to map claims to URLs found in that text
        prompt = f"""
        Compare the BLOG CLAIMS against the RESEARCH DATA.
        
        RESEARCH DATA (contains URLs):
        {research_data[:4000]}
        
        BLOG EXCERPT:
        {blog_post[:2000]}
        
        TASK:
        Identify 5 factual claims in the blog. For each:
        1. Categorize: [SUPPORTED], [NEUTRAL], or [CONTRADICTED].
        2. Provide the Source URL from the RESEARCH DATA that justifies this category.
        
        Format:
        CLAIM X: [CATEGORY]
        SOURCE: [URL or "None"]
        REASON: [Short explanation]
        """
        try:
            res = llm.invoke(prompt).content
            # Score logic remains the same
            sup = res.count("[SUPPORTED]")
            con = res.count("[CONTRADICTED]")
            score = max(0, min(10, (sup * 2) - (con * 5)))
            
            return {
                "score": float(score),
                "hallucination_detected": con > 0,
                "report": res # This now contains the URLs
            }
        except:
            return {"score": 5, "report": "Error in NLI"}

# ============================================================================
# 4. EVALUATOR - Tier 3: Qualitative Feedback
# ============================================================================

class LLMFeedback:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def get_improvements(self, blog_post: str, topic: str, scores: Dict) -> Dict:
        prompt = f"Analyze blog on '{topic}'. Scores: {scores}. Provide 3 short tips."
        response = self.llm.invoke(prompt)
        return {"feedback": response.content}

# ============================================================================
# 5. MAIN EVALUATION FUNCTION
# ============================================================================

def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
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