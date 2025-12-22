import os
import re
import json
from typing import Tuple, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ============================================================================
# 1. TOPIC VALIDATOR - One-shot validation to save cost
# ============================================================================

class TopicValidator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0) # Temp 0 for consistency

    def _basic_syntax_check(self, topic: str) -> Tuple[bool, str]:
        topic = topic.strip()
        if len(topic) < 3: return False, "❌ Topic too short"
        if len(topic) > 200: return False, "❌ Topic too long"
        if not any(c.isalpha() for c in topic): return False, "❌ No words detected"
        if re.search(r'(.)\1{4,}', topic): return False, "❌ Gibberish detected"
        return True, "OK"

    def _llm_gatekeeper(self, topic: str) -> Tuple[bool, str]:
        """Consolidated semantic, safety, and feasibility check."""
        prompt = PromptTemplate(
            template="""Evaluate the blog topic: "{topic}"
            Return ONLY a JSON object with:
            {{
                "valid": boolean,
                "reason": "Clear explanation of why it failed or 'Approved'"
            }}
            
            Criteria:
            1. Coherence: Is it a real, understandable concept?
            2. Safety: No instructions for violence, illegal acts, or self-harm.
            3. Researchability: Is it possible to find factual data about this? (No 'how to teleport')
            """,
            input_variables=["topic"],
        )
        try:
            chain = prompt | self.llm
            response = chain.invoke({"topic": topic})
            # Clean response content for JSON parsing
            clean_content = response.content.replace("```json", "").replace("```", "").strip()
            res = json.loads(clean_content)
            return res.get("valid", False), res.get("reason", "Unknown validation error")
        except Exception as e:
            return False, f"Validation system error: {str(e)}"

    def validate(self, topic: str) -> Dict:
        # Layer 1: Regex/Syntax
        ok, msg = self._basic_syntax_check(topic)
        if not ok: return {"valid": False, "reason": msg}

        # Layer 2: LLM Gatekeeper
        ok, msg = self._llm_gatekeeper(topic)
        return {"valid": ok, "reason": msg}


# ============================================================================
# 2. EVALUATOR - Fixed Math and Heuristics
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
        # Simplified Flesch for logic (Note: textstat library is better for production)
        words = blog_post.split()
        sentences = [s for s in re.split(r'[.!?]+', blog_post) if s.strip()]
        if not words or not sentences: return {"score": 0, "details": "Empty content"}
        
        avg_sentence_len = len(words) / len(sentences)
        # Penalize if sentences are too long (standard blog readability)
        score = 10 - (max(0, avg_sentence_len - 20) / 2)
        return {"score": max(0, round(score, 1)), "details": f"Avg Sentence Length: {avg_sentence_len:.1f}"}

class DataDrivenMetrics:
    @staticmethod
    def calculate_research_overlap(blog_post: str, research_data: str) -> Dict:
        """Measure keyword overlap (NOT fact-checking)."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', blog_post) if len(s.strip()) > 15]
        research_lower = research_data.lower()
        verified = 0
        
        for s in sentences:
            words = set(re.findall(r'\b[a-z]{5,}\b', s.lower())) # Keywords 5+ chars
            if words and sum(1 for w in words if w in research_lower) >= 2:
                verified += 1
        
        rate = (verified / len(sentences)) if sentences else 0
        return {"score": round(rate * 10, 1), "details": f"{int(rate*100)}% of sentences contain research keywords"}

# ============================================================================
# 3. MAIN EVALUATION FUNCTION
# ============================================================================

def realistic_evaluation(blog_post: str, research_data: str, topic: str) -> Dict:
    # Tier 1: Logic
    structure = HardMetrics.calculate_structure(blog_post)
    readability = HardMetrics.calculate_readability(blog_post)
    
    # Tier 2: Research
    overlap = DataDrivenMetrics.calculate_research_overlap(blog_post, research_data)
    
    # CALCULATE WEIGHTED SCORE (No magic '+5' math error)
    tier1_score = (structure['score'] + readability['score']) / 2
    tier2_score = overlap['score']
    
    final_score = (tier1_score * 0.5) + (tier2_score * 0.5)
    
    # Tier 3: Qualitative (LLM Feedback - Not included in math)
    feedback_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    prompt = f"Analyze this blog on '{topic}'. Structure Score: {structure['score']}, Research Score: {overlap['score']}. Provide 3 short improvements."
    feedback = feedback_llm.invoke(prompt).content

    return {
        "final_score": round(final_score, 1),
        "tier1": {"structure": structure, "readability": readability},
        "tier2": {"overlap": overlap},
        "tier3": {"feedback": feedback}
    }
