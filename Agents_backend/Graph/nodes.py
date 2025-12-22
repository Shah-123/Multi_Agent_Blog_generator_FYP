"""
nodes.py

This module defines all agent nodes used in the multi-agent blog generation
pipeline. Each node performs a single, well-defined responsibility and
operates on a shared AgentState object.

Nodes included:
1. Researcher Node      – Gathers and synthesizes external information
2. Analyst Node         – Converts research into an SEO-optimized outline
3. Writer Node          – Produces a full blog post from the outline
4. Fact-Checker Node    – Verifies claims against original research

All prompt templates are imported from Graph.templates to maintain separation
of concerns and improve maintainability.
"""

import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.exceptions import LangChainException

import os
from typing import Dict, Any, List
from urllib.parse import urlparse
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI

from Graph.state import AgentState
from Graph.templates import (
    RESEARCHER_PROMPT,
    ANALYST_PROMPT,
    WRITER_PROMPT,
    FACT_CHECKER_PROMPT,
)

# ---------------------------------------------------------------------------
# LLM & TOOL CONFIGURATION
# ---------------------------------------------------------------------------

tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

tavily_tool = TavilySearchResults(
    max_results=10,
    tavily_api_key=tavily_api_key,
)

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
)

# ============= ADD THIS ENTIRE SECTION =============

# TRUSTED SOURCES - These are always good
TRUSTED_DOMAINS = {
    # News
    "bbc.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "theverge.com",
    "techcrunch.com",
    
    # Academic
    "arxiv.org",
    "scholar.google.com",
    "nature.com",
    
    # Educational
    "wikipedia.org",
    "britannica.com",
    "investopedia.com",
    "khan-academy.org",
    
    # Government
    "gov.uk",
    "state.gov",
    ".edu",
    ".gov",
}

# BLOCKED SOURCES - Never use these
BLOCKED_DOMAINS = {
    "reddit.com",
    "quora.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "medium.com",
    "tumblr.com",
    "blogspot.com",
    "wordpress.com",
}






def validate_source(result: dict) -> tuple[bool, str]:
    """
    Check if a source is good quality.
    Returns: (is_valid: bool, reason: str)
    """
    url = result.get("url", "").lower()
    title = result.get("title", "").lower()
    content = result.get("content", "").lower()
    
    # Extract domain
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
    except:
        return False, "Invalid URL format"
    
    # CHECK 1: Blocked domains - automatic rejection
    for blocked in BLOCKED_DOMAINS:
        if blocked in domain:
            return False, f"Blocked domain"
    
    # CHECK 2: Trusted domains - automatic accept
    for trusted in TRUSTED_DOMAINS:
        if trusted in domain:
            return True, f"Trusted source"
    
    # CHECK 3: Educational/Government - good
    if domain.endswith(".edu") or domain.endswith(".gov") or domain.endswith(".org"):
        return True, "Official organization"
    
    # CHECK 4: Content too short = snippet only
    if len(content) < 200:
        return False, "Content too short"
    
    # CHECK 5: Opinion pieces
    opinion_keywords = ["i think", "my opinion", "imho", "fake news"]
    for keyword in opinion_keywords:
        if keyword in content[:500]:
            return False, "Opinion piece"
    
    # CHECK 6: AI-generated markers
    ai_markers = ["as an ai", "i'm an ai", "i cannot"]
    for marker in ai_markers:
        if marker in content[:500]:
            return False, "AI-generated content"
    
    return True, "Quality source"


def generate_alternative_searches(topic: str) -> List[str]:
    """
    If first search fails, try different search terms.
    """
    return [
        f"{topic} 2024",
        f"{topic} latest",
        f"{topic} research",
        f"{topic} study",
    ]
# ---------------------------------------------------------------------------
# NODE 1: RESEARCHER
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Node with Source Validation
    """
    try:
        topic = state.get("topic", "").strip()
        if not topic:
            return {"error": "Topic cannot be empty"}

        print("--- RESEARCHER AGENT WORKING ---")
        print(f"Topic: {topic}")
        print(f"🔍 Searching...")

        # Get raw results
        search_results = tavily_tool.invoke(topic)
        if not search_results:
            return {"error": "No search results found"}

        print(f"📊 Tavily returned {len(search_results)} results")

        # Validate each result
        validated_sources = []
        rejected_sources = []

        for i, result in enumerate(search_results, 1):
            is_valid, reason = validate_source(result)

            if is_valid:
                validated_sources.append({
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "source": result.get("source"),
                    "content": result.get("content"),
                    "reason": reason,
                })
                print(f"  ✓ [{i}] {result.get('title')[:50]}...")
            else:
                rejected_sources.append({
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "reason": reason,
                })
                print(f"  ✗ [{i}] {result.get('title')[:50]}... ({reason})")

        print(f"\n📈 Valid: {len(validated_sources)}, Rejected: {len(rejected_sources)}")

        # If not enough good sources, retry
        MIN_SOURCES = 3
        if len(validated_sources) < MIN_SOURCES:
            print(f"⚠️  Only {len(validated_sources)} sources. Trying alternatives...\n")

            alternative_searches = generate_alternative_searches(topic)

            for alt_topic in alternative_searches:
                print(f"🔄 Retry: '{alt_topic}'")
                alt_results = tavily_tool.invoke(alt_topic)

                for result in alt_results:
                    is_valid, reason = validate_source(result)
                    if is_valid:
                        validated_sources.append({
                            "url": result.get("url"),
                            "title": result.get("title"),
                            "source": result.get("source"),
                            "content": result.get("content"),
                            "reason": reason,
                        })
                        print(f"  ✓ {result.get('title')[:50]}...")

                if len(validated_sources) >= MIN_SOURCES:
                    break

        # Not enough sources
        if len(validated_sources) < MIN_SOURCES:
            return {
                "error": f"Only found {len(validated_sources)} valid sources. Quality may be low."
            }

        # Build research from validated sources only
        search_content = ""
        sources_for_state = []

        for source in validated_sources:
            search_content += (
                f"Source: {source.get('url')}\n"
                f"Title: {source.get('title', 'N/A')}\n"
                f"Content: {source.get('content')}\n\n"
            )

            sources_for_state.append({
                "url": source.get("url"),
                "title": source.get("title"),
                "source": source.get("source"),
                "validation_status": source.get("reason"),
            })

        # Pass to GPT
        chain = RESEARCHER_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "search_content": search_content,
        })

        print(f"✅ Research complete using {len(validated_sources)} sources\n")

        return {
            "research_data": response.content,
            "sources": sources_for_state,
        }

    except Exception as e:
        return {"error": f"Researcher error: {str(e)}"}
# ---------------------------------------------------------------------------
# NODE 2: ANALYST
# ---------------------------------------------------------------------------

def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyst Node

    Purpose:
        Transforms structured research data into a comprehensive,
        SEO-optimized blog outline.

    Inputs (AgentState):
        - topic (str)
        - research_data (str)

    Outputs (AgentState updates):
        - blog_outline (str): Markdown-formatted blog outline
        - error (str, optional)
    """
    try:
        if state.get("error"):
            return {"error": f"Cannot analyze due to prior error: {state['error']}"}

        topic = state.get("topic")
        research_data = state.get("research_data")

        if not research_data:
            return {"error": "No research data available for analysis"}

        print("--- ANALYST AGENT WORKING ---")
        print(f"Creating outline for: {topic}")

        chain = ANALYST_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "research_data": research_data,
        })

        return {"blog_outline": response.content}

    except LangChainException as e:
        return {"error": f"Analyst node failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected analyst error: {str(e)}"}


# ---------------------------------------------------------------------------
# NODE 3: WRITER
# ---------------------------------------------------------------------------

def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Writer Node

    Purpose:
        Generates a complete, publication-ready blog post using the
        provided outline and research data.

    Inputs (AgentState):
        - topic (str)
        - research_data (str)
        - blog_outline (str)

    Outputs (AgentState updates):
        - final_blog_post (str): Full blog post in Markdown
        - error (str, optional)
    """
    try:
        if state.get("error"):
            return {"error": f"Cannot write due to prior error: {state['error']}"}

        topic = state.get("topic")
        research_data = state.get("research_data")
        blog_outline = state.get("blog_outline")

        if not blog_outline:
            return {"error": "No blog outline available"}
        if not research_data:
            return {"error": "No research data available"}

        print("--- WRITER AGENT WORKING ---")
        print(f"Writing blog post for: {topic}")

        chain = WRITER_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "blog_outline": blog_outline,
            "research_data": research_data,
        })

        return {"final_blog_post": response.content}

    except LangChainException as e:
        return {"error": f"Writer node failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected writer error: {str(e)}"}


# ---------------------------------------------------------------------------
# NODE 4: FACT-CHECKER
# ---------------------------------------------------------------------------

def fact_checker_node(state: AgentState) -> Dict[str, Any]:
    """
    Fact-Checker Node

    Purpose:
        Verifies factual accuracy of the generated blog post by
        cross-checking claims against original research data and sources.

    Inputs (AgentState):
        - topic (str)
        - research_data (str)
        - final_blog_post (str)
        - sources (list[dict])

    Outputs (AgentState updates):
        - fact_check_report (str): Detailed verification report
        - error (str, optional)
    """
    try:
        if state.get("error"):
            return {"error": f"Cannot fact-check due to prior error: {state['error']}"}

        topic = state.get("topic")
        research_data = state.get("research_data")
        blog_post = state.get("final_blog_post")
        sources = state.get("sources", [])

        if not blog_post:
            return {"error": "No blog post available for fact-checking"}
        if not research_data:
            return {"error": "No research data available for verification"}

        print("--- FACT-CHECKER AGENT WORKING ---")
        print("Verifying blog post claims...")

        sources_info = "\n".join(
            f"- {s.get('title')} ({s.get('url')})"
            for s in sources
        ) if sources else "No sources provided"

        chain = FACT_CHECKER_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "research_data": research_data,
            "blog_post": blog_post,
            "sources_info": sources_info,
        })

        return {"fact_check_report": response.content}

    except LangChainException as e:
        return {"error": f"Fact-checker node failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected fact-checker error: {str(e)}"}



def regenerate_blog_with_feedback(blog_outline: str, research_data: str, topic: str, llm_feedback: str, iteration: int = 1) -> str:
    """
    Regenerate blog post using LLM feedback to improve content.
    
    Args:
        blog_outline: Original blog outline
        research_data: Research data used
        topic: Blog topic
        llm_feedback: Feedback from LLM about improvements needed
        iteration: Current iteration number
    
    Returns:
        Improved blog post
    """
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    prompt = f"""You are an expert content writer. A blog post has been evaluated and here's the feedback:

FEEDBACK FOR IMPROVEMENT:
{llm_feedback}

TOPIC: {topic}
ORIGINAL OUTLINE:
{blog_outline}

RESEARCH DATA:
{research_data}

Your task: Rewrite the blog post to address ALL the feedback points mentioned above.
- Improve specific areas mentioned in the feedback
- Maintain the original structure/outline
- Keep all factual accuracy
- Enhance readability and engagement
- Make it publication-ready

Write the complete blog post in Markdown format with proper headings, sections, and formatting."""
    
    response = llm.invoke(prompt)
    improved_blog = response.content
    
    print(f"\n✅ Blog regenerated (Iteration {iteration})")
    print(f"📏 New length: {len(improved_blog)} characters")
    
    return improved_blog