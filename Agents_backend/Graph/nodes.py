import os
from typing import Dict, Any, List
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.exceptions import LangChainException


import time
import requests
import io
from PIL import Image
import uuid

# --- Add these to your nodes.py ---

from Graph.state import AgentState
from Graph.templates import (
    RESEARCHER_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    FACT_CHECKER_PROMPT,
)

# HF_TOKEN = os.getenv("HF_TOKEN")
# # You can change this to "black-forest-labs/FLUX.1-schnell" or "stabilityai/stable-diffusion-xl-base-1.0"
# API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
# headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

tavily_tool = TavilySearchResults(max_results=8, tavily_api_key=tavily_api_key)
llm = ChatOpenAI(model="gpt-4o", temperature=0.5)

TRUSTED_DOMAINS = {
    "bbc.com", "reuters.com", "apnews.com", "theguardian.com", "theverge.com", 
    "techcrunch.com", "arxiv.org", "nature.com", "wikipedia.org", "britannica.com",
    "investopedia.com", "gov.uk", "state.gov"
}

BLOCKED_DOMAINS = {
    "reddit.com", "quora.com", "twitter.com", "x.com", "facebook.com", 
    "instagram.com", "tiktok.com", "medium.com", "blogspot.com"
}

# ---------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def validate_source(result: dict) -> tuple[bool, str]:
    """Basic source quality gatekeeper."""
    url = result.get("url", "").lower()
    content = result.get("content", "").lower()
    
    try:
        domain = urlparse(url).netloc.replace("www.", "")
    except:
        return False, "Invalid URL"

    if any(blocked in domain for blocked in BLOCKED_DOMAINS):
        return False, "Unreliable/Social Media"
    
    if any(trusted in domain for trusted in TRUSTED_DOMAINS) or domain.endswith((".edu", ".gov", ".org")):
        return True, "Trusted Source"
    
    if len(content) < 300:
        return False, "Insufficient content"
        
    return True, "General Web"

# ---------------------------------------------------------------------------
# 3. GRAPH NODES
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Gathers and filters research data."""
    topic = state.get("topic", "").strip()
    print(f"--- 🔍 RESEARCHING: {topic} ---")

    try:
        raw_results = tavily_tool.invoke(topic)
        validated_sources = []
        source_metadata = []

        for res in raw_results:
            is_valid, reason = validate_source(res)
            if is_valid:
                validated_sources.append(res)
                source_metadata.append({
                    "url": res.get("url"),
                    "title": res.get("title"),
                    "reason": reason
                })

        if len(validated_sources) < 2:
            return {"error": "Insufficient quality sources found."}

        # Combine content for LLM synthesis
        context = "\n\n".join([f"Source: {s['url']}\nContent: {s['content'][:1500]}" for s in validated_sources])
        
        chain = RESEARCHER_PROMPT | llm
        response = chain.invoke({"topic": topic, "search_content": context})

        return {
            "research_data": response.content,
            "sources": source_metadata
        }
    except Exception as e:
        return {"error": f"Researcher failed: {str(e)}"}

def analyst_node(state: AgentState) -> Dict[str, Any]:
    """Creates SEO outline."""
    if state.get("error"): return {}
    print("--- 📋 ANALYZING ---")
    
    try:
        chain = ANALYST_PROMPT | llm
        res = chain.invoke({"topic": state["topic"], "research_data": state["research_data"]})
        return {"blog_outline": res.content}
    except Exception as e:
        return {"error": f"Analyst failed: {str(e)}"}

def writer_node(state: AgentState) -> Dict[str, Any]:
    if state.get("error"): return {}
    print("--- ✍️ WRITING ---")
    
    # Get feedback from previous evaluation if it exists
    feedback = ""
    if state.get("quality_evaluation"):
        feedback = state["quality_evaluation"].get("tier3", {}).get("feedback", "")

    try:
        chain = WRITER_PROMPT | llm
        res = chain.invoke({
            "topic": state["topic"], 
            "blog_outline": state["blog_outline"], 
            "research_data": state["research_data"],
            "feedback": feedback  # PASS THE FEEDBACK HERE
        })
        return {"final_blog_post": res.content}
    except Exception as e:
        return {"error": f"Writer failed: {str(e)}"}

def fact_checker_node(state: AgentState) -> Dict[str, Any]:
    """Checks for hallucinations."""
    if state.get("error"): return {}
    print("--- 🔍 FACT-CHECKING ---")
    
    try:
        chain = FACT_CHECKER_PROMPT | llm
        res = chain.invoke({
            "topic": state["topic"],
            "research_data": state["research_data"],
            "blog_post": state["final_blog_post"],
            "sources_info": str(state["sources"])
        })
        return {"fact_check_report": res.content}
    except Exception as e:
        return {"error": f"Fact-checker failed: {str(e)}"}

# ---------------------------------------------------------------------------
# 4. REGENERATION LOGIC (Used by post-processing)
# ---------------------------------------------------------------------------

def regenerate_blog_with_feedback(state_data: dict, feedback: str) -> str:
    """Uses evaluation feedback to rewrite the blog."""
    print("--- 🔄 RE-WRITING WITH FEEDBACK ---")
    
    prompt = f"""
    Rewrite the following blog post based on this feedback:
    FEEDBACK: {feedback}
    
    ORIGINAL BLOG:
    {state_data.get('final_blog_post')}
    
    TOPIC: {state_data.get('topic')}
    RESEARCH: {state_data.get('research_data')}
    """
    
    res = llm.invoke(prompt)
    return res.content




# def image_generator_node(state: AgentState) -> Dict[str, Any]:
#     """Generates a featured image with retry logic for Hugging Face."""
#     print("--- 🎨 GENERATING FEATURED IMAGE ---")
    
#     topic = state["topic"]
    
#     # 1. Generate the Prompt
#     prompt_chain = ChatOpenAI(model="gpt-4o", temperature=0.7)
#     image_prompt_res = prompt_chain.invoke(f"Create a high-quality, professional digital art prompt for a blog titled '{topic}'. No text in image.")
#     refined_prompt = image_prompt_res.content

#     # 2. API Call with Retry Logic
#     max_retries = 3
#     for attempt in range(max_retries):
#         try:
#             print(f"🖌️ Attempt {attempt + 1}: Querying Hugging Face...")
#             response = requests.post(API_URL, headers=headers, json={"inputs": refined_prompt})
            
#             # Check if API returned an error (like model loading)
#             if response.status_code != 200:
#                 error_data = response.json()
#                 if "estimated_time" in error_data:
#                     wait_time = error_data["estimated_time"]
#                     print(f"⏳ Model is loading. Waiting {wait_time:.1f}s...")
#                     time.sleep(min(wait_time, 20)) # Wait but cap it at 20s
#                     continue # Try again
#                 else:
#                     print(f"❌ API Error: {error_data}")
#                     break

#             # 3. Try to process the image bytes
#             image_bytes = response.content
#             image = Image.open(io.BytesIO(image_bytes))
            
#             # 4. Save
#             file_path = f"blog_image_{uuid.uuid4().hex[:8]}.png"
#             image.save(file_path)
#             print(f"✅ Image saved successfully: {file_path}")
            
#             # Inject into blog
#             updated_blog = f"![Featured Image]({file_path})\n\n{state['final_blog_post']}"
#             return {
#                 "image_paths": [file_path],
#                 "final_blog_post": updated_blog
#             }

#         except Exception as e:
#             print(f"⚠️ Attempt {attempt + 1} failed: {e}")
#             time.sleep(5)

#     print("❌ Failed to generate image after retries.")
#     return {"error": "Image generation failed."}