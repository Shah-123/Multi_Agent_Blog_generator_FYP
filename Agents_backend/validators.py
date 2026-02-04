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

# # FIXED: Add basic tests at bottom
# if __name__ == "__main__":
#     # Quick test
#     test_blog = """# Emerging Jobs with AI: Opportunities for 2026 and Beyond

# The rapid evolution of artificial intelligence (AI) is not just a technological phenomenon; it is reshaping the job landscape at an unprecedented pace. According to recent studies, the demand for AI-related jobs is projected to grow by **40% by 2026**, highlighting a significant shift in employment opportunities within the sector [Source](https://www.lorienglobal.com/insights/emerging-ai-jobs-in-demand). This surge in demand raises a critical question: will AI lead to job displacement, or will it create new opportunities that can absorb the workforce?

# As organizations increasingly adopt AI technologies, the potential for job displacement looms large. Traditional roles may become obsolete, leaving many workers vulnerable. However, this disruption also paves the way for the emergence of new job categories that require specialized skills. The challenge lies in ensuring that the workforce is equipped to transition into these roles, thereby mitigating the adverse effects of automation.

# ### Key Emerging Roles in AI

# Several key positions are expected to dominate the AI job market by 2026. First, **AI Ethicists** will play a crucial role in addressing the ethical implications of AI technologies, ensuring that they are developed and implemented responsibly. Second, **Machine Learning Engineers** will be in high demand as they design and optimize algorithms that enable machines to learn from data. Additionally, **Data Curators** will emerge as vital players, tasked with managing and organizing the vast amounts of data that fuel AI systems. Finally, **AI Trainers** will be essential for teaching AI systems to understand and interpret human behavior, bridging the gap between technology and user experience.

# In summary, while the rise of AI presents challenges related to job displacement, it simultaneously offers a wealth of opportunities for those willing to adapt. The future workforce must be prepared to embrace these emerging roles, ensuring that they remain relevant in an increasingly automated world.

# The rapid evolution of artificial intelligence (AI) is reshaping the job landscape, creating a myriad of opportunities for professionals across various sectors. **Artificial Intelligence** refers to the simulation of human intelligence processes by machines, particularly computer systems. This encompasses **machine learning**, a subset of AI that enables systems to learn from data and improve their performance over time, and **automation**, which involves using technology to perform tasks with minimal human intervention. Understanding these terms is crucial as they form the foundation of the emerging job market.

# ### Current Job Market for AI Professionals

# As of now, the demand for AI professionals is surging. According to recent insights, the AI job market is projected to grow significantly, with roles such as AI engineers, data scientists, and machine learning specialists becoming increasingly vital. A report from LinkedIn highlights that new AI roles are expected to emerge by 2026, reflecting the industry's need for skilled individuals who can navigate complex AI systems and contribute to innovative solutions [LinkedIn](https://www.linkedin.com/pulse/new-ai-roles-2026-anastasiia-shapovalova-wigmc). Furthermore, a study indicates that entry-level positions in AI are also on the rise, making this an opportune time for newcomers to enter the field [Study.com](https://study.com/resources/top-entry-level-ai-jobs.html).

# ### The Importance of AI Jobs in Today's Economy

# AI jobs are not merely a trend; they are critical to the modern economy. As businesses increasingly rely on AI to enhance efficiency and drive innovation, the need for skilled professionals who can develop, implement, and manage AI technologies becomes paramount. The integration of AI into various industries is expected to create new job categories and transform existing roles, thereby contributing to economic growth. For instance, a report by Salesforce emphasizes that AI is not only creating jobs but also enhancing productivity across sectors, which is essential for maintaining competitive advantage in a rapidly changing market [Salesforce](https://www.salesforce.com/blog/ai-jobs/).

# In summary, the current state of AI jobs reflects a dynamic and expanding landscape that is crucial for economic development. As we look toward 2026 and beyond, the importance of understanding AI and its implications for the job market cannot be overstated. What skills will you need to thrive in this evolving environment?

# As artificial intelligence continues to evolve, so too does the job market, giving rise to a variety of specialized roles that address the unique challenges and opportunities presented by this technology. By 2026, several key positions are expected to emerge as critical components of the AI landscape.

# ### AI Security and Risk Analyst

# One of the most vital roles will be that of the **AI Security and Risk Analyst**. This position focuses on identifying and mitigating risks associated with AI systems, ensuring that organizations can leverage AI technologies without compromising security. Responsibilities include conducting risk assessments, developing security protocols, and monitoring AI systems for vulnerabilities. As AI becomes more integrated into business operations, the demand for professionals who can safeguard these systems will increase significantly. According to a report by [Lorien Insights](https://www.lorienglobal.com/insights/emerging-ai-jobs-in-demand), the need for such analysts is projected to grow as organizations prioritize data protection and compliance.

# ### Prompt Engineer

# Another emerging role is that of the **Prompt Engineer**, a position that has gained prominence with the rise of conversational AI and natural language processing. Prompt Engineers are responsible for designing and refining the prompts that guide AI models in generating accurate and contextually relevant responses. Their work is crucial in enhancing user experience and ensuring that AI systems understand and respond appropriately to human input. This role not only requires technical expertise in AI but also a deep understanding of language and communication. As highlighted in a recent article on [Salesforce](https://www.salesforce.com/blog/ai-jobs/), the importance of Prompt Engineers will only grow as businesses seek to improve their AI interactions.

# ### AI Ethicist and Algorithm Bias Auditor

# Lastly, the role of the **AI Ethicist** and **Algorithm Bias Auditor** is becoming increasingly essential. These professionals are tasked with ensuring that AI systems operate fairly and ethically, addressing concerns related to bias and discrimination in algorithms. Their responsibilities include auditing AI models for bias, developing ethical guidelines, and advocating for responsible AI practices within organizations. The demand for these roles is driven by a growing awareness of the societal implications of AI technologies. As noted in the article from [LinkedIn](https://www.linkedin.com/pulse/new-ai-roles-2026-anastasiia-shapovalova-wigmc), organizations are recognizing the need for ethical oversight to maintain public trust and comply with regulatory standards.

# In summary, the landscape of AI jobs in 2026 will be shaped by the need for security, effective communication, and ethical considerations. As these roles develop, they will play a crucial part in guiding the responsible integration of AI into various sectors.

# As artificial intelligence (AI) continues to evolve, the demand for skilled professionals in this field is surging. To thrive in emerging AI careers, individuals must cultivate a blend of technical and interpersonal skills that align with industry needs.

# ### Technical Skills: The Foundation of AI Careers

# At the core of any AI role are essential **technical skills**. Proficiency in **data analysis** is paramount, as it enables professionals to interpret complex datasets and derive actionable insights. Familiarity with programming languages such as Python, R, and Java is also critical, as these languages are commonly used in AI development and machine learning applications. According to a report on emerging AI jobs, candidates with strong programming skills are more likely to secure positions in this competitive landscape [Lorien Insights](https://www.lorienglobal.com/insights/emerging-ai-jobs-in-demand).

# Moreover, understanding machine learning algorithms and frameworks, such as TensorFlow and PyTorch, is increasingly important. These tools facilitate the development of AI models that can learn from data and improve over time. As the industry progresses, the ability to work with cloud computing platforms, such as AWS and Azure, will also become a valuable asset, allowing professionals to deploy AI solutions at scale.

# ### The Importance of 'Power Skills'

# While technical expertise is crucial, the significance of **'power skills'**—such as collaboration and communication—cannot be overstated. AI projects often involve cross-functional teams, requiring individuals to effectively convey complex ideas to non-technical stakeholders. Strong communication skills foster collaboration, ensuring that diverse perspectives are integrated into AI solutions. A study highlights that professionals who excel in teamwork and interpersonal communication are more likely to succeed in AI roles [Salesforce](https://www.salesforce.com/blog/ai-jobs/).

# Additionally, adaptability and problem-solving abilities are essential in a field characterized by rapid change. Professionals must be willing to learn continuously and pivot as new technologies emerge. This adaptability not only enhances individual performance but also contributes to the overall success of AI initiatives within organizations.

# ### Certifications and On-the-Job Training

# To further enhance their qualifications, aspiring AI professionals should consid... this is what i get and btw why at the end the sentence is not completed ."""
    
#     evaluator = BlogEvaluator()
#     result = evaluator.evaluate(test_blog, "Test Blog")
#     print(f"Test score: {result['final_score']}/10")
#     print(f"Verdict: {result['verdict']}")