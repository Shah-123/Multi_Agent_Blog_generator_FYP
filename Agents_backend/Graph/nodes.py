import os
import json
import re
import uuid
import requests
from typing import Dict, Any, List
from urllib.parse import urlparse

# 1. IMPORTS
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
# 🆕 OFFICIAL HUGGING FACE CLIENT
from huggingface_hub import InferenceClient 

from Graph.state import AgentState
from Graph.templates import (
    RESEARCHER_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    FACT_CHECKER_PROMPT,
    COMPETITOR_ANALYSIS_PROMPT
)

# ---------------------------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------------------------

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
llm_cheap = ChatOpenAI(model="gpt-4o-mini", temperature=0)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/scrape"

BLOCKED_DOMAINS = {
    "reddit.com", "quora.com", "twitter.com", "x.com", "facebook.com", 
    "instagram.com", "tiktok.com", "medium.com", "blogspot.com"
}

# ---------------------------------------------------------------------------
# 🆕 3. RESEARCH COMPRESSOR (Token Leak Fix)
# ---------------------------------------------------------------------------

class ResearchCompressor:
    """Compress 16KB research → 2KB structured data."""
    
    def __init__(self):
        self.llm = llm_cheap
    
    def compress(self, raw_research: str, topic: str) -> Dict[str, Any]:
        """Convert raw research into structured, compact format."""
        
        prompt = PromptTemplate(
            template="""
Extract key facts from this research on '{topic}'. Be RUTHLESS about compression.

RAW RESEARCH:
{raw_research}

OUTPUT AS PURE JSON (NO OTHER TEXT):
{{
    "key_facts": [
        {{"fact": "specific factual claim", "url": "source URL", "confidence": "HIGH/MEDIUM/LOW"}},
        {{...max 5 facts}}
    ],
    "statistics": [
        {{"stat": "specific number/percentage", "context": "what it refers to", "url": "source URL", "year": 2024}},
        {{...max 5 stats}}
    ],
    "quotes": [
        {{"quote": "exact quote under 50 words", "author": "name if given", "url": "source URL"}},
        {{...max 3 quotes}}
    ]
}}

CRITICAL RULES:
- Extract ONLY verifiable claims with URLs
- Maximum 5 facts, 5 statistics, 3 quotes total
- No fluff or generic statements
- If fact has no URL, EXCLUDE IT
- Each entry MUST be citable
""",
            input_variables=["topic", "raw_research"],
        )
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "topic": topic,
                "raw_research": raw_research[:12000]
            })
            
            json_text = response.content
            json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"⚠️ Compression failed: {e}")
        
        return {
            "key_facts": [],
            "statistics": [],
            "quotes": []
        }

# ---------------------------------------------------------------------------
# 🆕 4. CITATION INDEX (Token Leak Fix)
# ---------------------------------------------------------------------------

class CitationIndex:
    """Lightweight lookup for citations."""
    
    def __init__(self, compressed_research: Dict[str, Any]):
        self.compressed = compressed_research
    
    def to_string(self) -> str:
        """Convert to compact string format for LLM input (max 2KB)."""
        lines = []
        lines.append("# CITABLE CLAIMS INDEX\n")
        
        lines.append("## Facts")
        for f in self.compressed.get("key_facts", [])[:5]:
            lines.append(f"- {f['fact']} ({f['url']}) [Confidence: {f.get('confidence', 'MEDIUM')}]")
        
        lines.append("\n## Statistics")
        for s in self.compressed.get("statistics", [])[:5]:
            lines.append(f"- {s['stat']} ({s['url']}) [Year: {s.get('year', 'N/A')}]")
        
        lines.append("\n## Direct Quotes")
        for q in self.compressed.get("quotes", [])[:3]:
            author = f"- {q.get('author', 'Unknown')}" if q.get('author') else ""
            lines.append(f"- \"{q['quote']}\" {author} ({q['url']})")
        
        return "\n".join(lines)

# ---------------------------------------------------------------------------
# 5. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def validate_source(result: dict) -> tuple[bool, str]:
    url = result.get("url", "").lower()
    content = result.get("content", "").lower()
    if any(blocked in url for blocked in BLOCKED_DOMAINS): return False, "Unreliable"
    if len(content) < 300: return False, "Thin content"
    return True, "Valid"

def scrape_with_firecrawl_api(url: str) -> str:
    if not FIRECRAWL_API_KEY: return ""
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"url": url, "formats": ["markdown"]}
    try:
        res = requests.post(FIRECRAWL_API_URL, headers=headers, json=payload, timeout=30)
        if res.status_code == 200:
            data = res.json()
            if data.get("success"): return data["data"].get("markdown", "")
        return ""
    except:
        return ""

def extract_json_from_response(content: str) -> dict:
    """Robustly extracts JSON from LLM output using Regex."""
    try:
        return json.loads(content)
    except:
        pass

    # Regex: Find content between ```json and ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
            
    # Regex: Find first '{' and last '}'
    match = re.search(r"(\{.*\})", content, re.DOTALL)
    if match:
        try: return json.loads(match.group(1))
        except: pass
            
    # Fallback
    return {
        "blog_outline": content, 
        "sections": ["Introduction", "Main Body", "Conclusion"],
        "seo_title": "Generated Blog Post",
        "meta_description": "A blog post generated by AI.",
        "target_keywords": []
    }

# ---------------------------------------------------------------------------
# 6. GRAPH NODES - RESEARCHER (WITH COMPRESSION)
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> Dict[str, Any]:
    print("--- 🔍 RESEARCHING ---")
    topic = state.get("topic", "")
    
    try:
        raw_results = tavily_tool.invoke(topic)
        urls = [r['url'] for r in raw_results if validate_source(r)[0]][:2]
        
        full_content = []
        for url in urls:
            content = scrape_with_firecrawl_api(url)
            if content: full_content.append(f"Source: {url}\n{content[:8000]}")
            
        combined = "\n\n".join(full_content)
        
        # Run researcher chain (full research)
        research_res = (RESEARCHER_PROMPT | llm).invoke({"topic": topic, "search_content": combined})
        comp_res = (COMPETITOR_ANALYSIS_PROMPT | llm).invoke({"topic": topic, "search_content": combined})
        
        raw_research_data = research_res.content
        
        # 🆕 COMPRESSION STEP
        print("--- 📊 COMPRESSING RESEARCH ---")
        compressor = ResearchCompressor()
        compressed_research = compressor.compress(raw_research_data, topic)
        
        # 🆕 CREATE CITATION INDEX
        citation_idx = CitationIndex(compressed_research)
        citation_index_str = citation_idx.to_string()
        
        print(f"✅ Compressed: {len(raw_research_data)} chars → {len(json.dumps(compressed_research))} chars")
        
        return {
            "research_data": raw_research_data,  # Keep for backward compatibility with fact_checker
            "raw_research_data": combined,       # Keep for evaluator
            "compressed_research": compressed_research,  # 🆕
            "citation_index": citation_index_str,        # 🆕
            "competitor_headers": comp_res.content
        }
    except Exception as e:
        return {"error": f"Researcher failed: {str(e)}"}

# ---------------------------------------------------------------------------
# 7. GRAPH NODES - ANALYST
# ---------------------------------------------------------------------------

def analyst_node(state: AgentState) -> Dict[str, Any]:
    print("--- 📋 PLANNING ---")
    plan = state.get("plan", "basic").lower() 
    
    try:
        chain = ANALYST_PROMPT | llm
        res = chain.invoke({
            "topic": state["topic"], 
            "research_data": state["research_data"],
            "competitor_headers": state["competitor_headers"],
            "plan": plan
        })
        
        data = extract_json_from_response(res.content)
        
        seo_data = {
            "title": data.get("seo_title", "Blog Post"),
            "description": data.get("meta_description", ""),
            "keywords": data.get("target_keywords", [])
        }
        
        return {
            "blog_outline": data.get("blog_outline", ""),
            "sections": data.get("sections", []),
            "seo_metadata": seo_data
        }
    except Exception as e:
        print(f"Analyst Error: {e}")
        return {
            "blog_outline": "Basic Outline", 
            "sections": ["Introduction", "Key Concepts", "Conclusion"]
        }

# ---------------------------------------------------------------------------
# 8. GRAPH NODES - WRITER (WITH CITATION INDEX)
# ---------------------------------------------------------------------------

def writer_node(state: AgentState) -> Dict[str, Any]:
    print("--- ✍️ WRITING ---")
    
    feedback = state.get("quality_evaluation", {}).get("tier3", {}).get("feedback", "")
    if feedback and state.get("final_blog_post"):
        print("--- 🔄 REWRITING ---")
        res = llm.invoke(f"Rewrite based on: {feedback}\n\n{state['final_blog_post']}")
        return {"final_blog_post": res.content}

    sections = state.get("sections", [])
    if not sections: return {"error": "No sections to write"}
    
    clean_sections = list(dict.fromkeys([s for s in sections if s]))
    full_content = []
    
    # 🆕 Use citation_index if available, fallback to research_data
    citation_index = state.get("citation_index", "")
    research_data = state.get("research_data", "")
    
    for i, title in enumerate(clean_sections):
        print(f"   ✍️ Section {i+1}/{len(clean_sections)}: {title}")
        prev = full_content[-1][-1000:] if full_content else ""
        
        chain = WRITER_PROMPT | llm
        
        # 🆕 Use citation_index if available
        data_to_pass = citation_index if citation_index else research_data
        
        res = chain.invoke({
            "topic": state["topic"],
            "section_title": title,
            "previous_content": prev,
            "research_data": data_to_pass,  # 🆕 Can be citation_index or research_data
            "tone": state.get("tone", "Professional"),
            "confidence_level": "High"
        })
        
        text = res.content
        if f"## {title}" not in text: text = f"## {title}\n\n{text}"
        full_content.append(text)

    return {"final_blog_post": "\n\n".join(full_content)}

# ---------------------------------------------------------------------------
# 9. GRAPH NODES - FACT CHECKER
# ---------------------------------------------------------------------------

def fact_checker_node(state: AgentState) -> Dict[str, Any]:
    print("--- 🔍 FACT CHECKING ---")
    try:
        # 🆕 Try to use compressed research, fallback to raw
        research = state.get("raw_research_data") or state.get("research_data")
        
        # If we have compressed research, convert to string for fact-checker
        if state.get("compressed_research"):
            research = json.dumps(state.get("compressed_research"), indent=2)
        
        chain = FACT_CHECKER_PROMPT | llm
        res = chain.invoke({
            "topic": state["topic"],
            "research_data": research,
            "blog_post": state["final_blog_post"]
        })
        return {"fact_check_report": res.content}
    except Exception as e:
        return {"error": f"Fact check failed: {str(e)}"}

# ---------------------------------------------------------------------------
# 10. GRAPH NODES - IMAGE GENERATOR (YOUR VERSION)
# ---------------------------------------------------------------------------

def image_generator_node(state: AgentState) -> Dict[str, Any]:
    print("--- 🎨 GENERATING IMAGE (FLUX.1) ---")
    
    topic = state.get("topic", "")
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("   ⚠️ No HF_TOKEN found. Skipping image.")
        return {"image_path": None}

    try:
        # Initialize Client
        client = InferenceClient("black-forest-labs/FLUX.1-schnell", token=hf_token)
        
        # Generate Prompt using LLM
        prompt_request = f"""
        Create a detailed, artistic image generation prompt for a blog post about: '{topic}'.
        Style: Modern, Minimalist, Tech-focused, Digital Art. NO TEXT.
        Output ONLY the prompt.
        """
        image_prompt = llm.invoke(prompt_request).content
        print(f"   🎨 Prompt: {image_prompt[:60]}...")

        # Generate Image
        image = client.text_to_image(image_prompt)
        
        # Save to disk
        filename = f"blog_image_{uuid.uuid4().hex[:8]}.png"
        image.save(filename)
        print(f"   ✅ Image saved: {filename}")
        
        return {"image_path": filename}

    except Exception as e:
        print(f"   ⚠️ Image Gen Failed: {e}")
        return {"image_path": None}