import os
import requests
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# 🆕 NEW IMPORTS FOR STRUCTURED OUTPUT
from pydantic import BaseModel, Field 

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from Graph.state import AgentState
from Graph.templates import (
    RESEARCHER_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    FACT_CHECKER_PROMPT,
    COMPETITOR_ANALYSIS_PROMPT
)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found")

tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_api_key)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_API_URL = "https://api.firecrawl.dev/v1/scrape"

BLOCKED_DOMAINS = {
    "reddit.com", "quora.com", "twitter.com", "x.com", "facebook.com", 
    "instagram.com", "tiktok.com", "medium.com", "blogspot.com"
}

TRUSTED_DOMAINS = {
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com", "theverge.com", 
    "techcrunch.com", "arxiv.org", "nature.com", "wikipedia.org", "britannica.com",
    "investopedia.com", "gov.uk", "state.gov"
}

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def validate_source(result: dict) -> tuple[bool, str]:
    url = result.get("url", "").lower()
    content = result.get("content", "").lower()
    
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except:
        return False, "Invalid URL"

    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        return False, "Unreliable/Social Media"
    
    if len(content) < 300:
        return False, "Insufficient content"
        
    return True, "General Web"

def scrape_with_firecrawl_api(url: str) -> str:
    """Direct API call to Firecrawl."""
    if not FIRECRAWL_API_KEY:
        print("⚠️ Firecrawl API Key missing.")
        return ""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}"
    }
    
    payload = {
        "url": url,
        "formats": ["markdown"]
    }

    try:
        response = requests.post(FIRECRAWL_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "data" in data:
                return data["data"].get("markdown", "")
            return ""
        else:
            print(f"⚠️ Firecrawl API Error {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        print(f"⚠️ Firecrawl Request Failed: {e}")
        return ""

# ---------------------------------------------------------------------------
# 3. SCHEMA DEFINITION (The Solution)
# ---------------------------------------------------------------------------

class BlogPlan(BaseModel):
    """The structured output we want from the Analyst."""
    blog_outline: str = Field(description="The full Markdown outline of the blog post")
    sections: List[str] = Field(description="A list of section headers (H2) to write one by one")

# ---------------------------------------------------------------------------
# 4. GRAPH NODES
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> Dict[str, Any]:
    topic = state.get("topic", "").strip()
    print(f"--- 🔍 RESEARCHING & SCRAPING (Direct API): {topic} ---")

    try:
        # 1. Get Search Results
        raw_results = tavily_tool.invoke(topic)
        valid_urls = []
        source_metadata = []
        
        for res in raw_results:
            is_valid, reason = validate_source(res)
            if is_valid:
                valid_urls.append(res['url'])
                source_metadata.append({"url": res.get("url"), "title": res.get("title"), "reason": reason})
        
        target_urls = list(set(valid_urls))[:2]
        
        if not target_urls:
             return {"error": "No valid URLs found to scrape."}

        print(f"--- 🕷️ SCRAPING URLS: {target_urls} ---")
        
        full_page_content = []
        
        # 2. Scrape loop
        for url in target_urls:
            markdown_text = scrape_with_firecrawl_api(url)
            
            if markdown_text:
                full_page_content.append(f"Source: {url}\nContent: {markdown_text[:8000]}")
            else:
                print(f"⚠️ Failed to get content for {url}")

        if not full_page_content:
            return {"error": "Firecrawl failed to scrape any content."}

        combined_content = "\n\n".join(full_page_content)

        # 3. Run LLM Chains
        research_chain = RESEARCHER_PROMPT | llm
        research_response = research_chain.invoke({"topic": topic, "search_content": combined_content})

        competitor_chain = COMPETITOR_ANALYSIS_PROMPT | llm
        competitor_response = competitor_chain.invoke({"topic": topic, "search_content": combined_content})

        print("--- 🕵️ COMPETITOR GAPS IDENTIFIED ---")

        return {
            "research_data": research_response.content,
            "competitor_headers": competitor_response.content, 
            "sources": source_metadata
        }
    except Exception as e:
        return {"error": f"Researcher failed: {str(e)}"}

def analyst_node(state: AgentState) -> Dict[str, Any]:
    """Uses Structured Output to guarantee valid Lists."""
    if state.get("error"): return {}
    print("--- 📋 ANALYZING & PLANNING (STRUCTURED) ---")
    
    try:
        # 🆕 MAGICAL FIX: .with_structured_output()
        # This forces the LLM to fill our Class structure, preventing parsing errors.
        structured_llm = llm.with_structured_output(BlogPlan)
        chain = ANALYST_PROMPT | structured_llm
        
        plan: BlogPlan = chain.invoke({
            "topic": state["topic"], 
            "research_data": state["research_data"],
            "competitor_headers": state["competitor_headers"]
        })
        
        return {
            "blog_outline": plan.blog_outline,
            "sections": plan.sections
        }
    except Exception as e:
        print(f"Analyst Error: {e}")
        return {"error": f"Analyst failed: {str(e)}"}

def writer_node(state: AgentState) -> Dict[str, Any]:
    """Recursive Writer: Writes section by section."""
    if state.get("error"): return {}
    print("--- ✍️ RECURSIVE WRITING START ---")
    
    # 1. Handle Feedback/Rewrite Mode
    feedback = state.get("quality_evaluation", {}).get("tier3", {}).get("feedback", "")
    if feedback and state.get("final_blog_post"):
        print("--- 🔄 REWRITING BASED ON FEEDBACK ---")
        prompt = f"Rewrite this blog based on feedback: {feedback}\n\nORIGINAL BLOG: {state['final_blog_post']}"
        res = llm.invoke(prompt)
        return {"final_blog_post": res.content}

    # 2. Get Sections
    raw_sections = state.get("sections", [])
    if not raw_sections:
        return {"error": "No sections found. Analyst failed."}

    # 🆕 SANITIZATION STEP: Deduplicate and Clean
    # This removes duplicates while keeping the order
    seen = set()
    cleaned_sections = []
    for s in raw_sections:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            cleaned_sections.append(s_clean)
            seen.add(s_clean)

    print(f"--- 🧹 Sections Cleaned: Reduced from {len(raw_sections)} to {len(cleaned_sections)} ---")

    research_data = state.get("research_data", "")
    topic = state.get("topic", "")
    
    full_content = []
    
    # WRITING LOOP (Using cleaned_sections)
    for i, section_title in enumerate(cleaned_sections):
        print(f"   ✍️ Writing Section {i+1}/{len(cleaned_sections)}: {section_title}...")
        
        # Keep context reasonable (last 1000 chars) to prevent token overflow
        previous_context = full_content[-1][-2000:] if full_content else "Start of the article."
        
        chain = WRITER_PROMPT | llm
        res = chain.invoke({
            "topic": topic,
            "section_title": section_title,
            "previous_content": previous_context,
            "research_data": research_data
        })
        
        section_content = res.content
        
        # Ensure Header exists
        if f"# {section_title}" not in section_content and f"## {section_title}" not in section_content:
            section_content = f"## {section_title}\n\n{section_content}"
            
        full_content.append(section_content)

    print("--- ✍️ WRITING COMPLETE ---")
    return {"final_blog_post": "\n\n".join(full_content)}

def fact_checker_node(state: AgentState) -> Dict[str, Any]:
    if state.get("error"): return {}
    print("--- 🔍 FACT-CHECKING ---")
    try:
        chain = FACT_CHECKER_PROMPT | llm
        res = chain.invoke({
            "topic": state["topic"],
            "research_data": state["research_data"],
            "blog_post": state["final_blog_post"],
        })
        return {"fact_check_report": res.content}
    except Exception as e:
        return {"error": f"Fact-checker failed: {str(e)}"}