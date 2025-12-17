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
    temperature=0,
)

llm_creative = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
)

# ---------------------------------------------------------------------------
# NODE 1: RESEARCHER
# ---------------------------------------------------------------------------

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Node

    Purpose:
        Performs external research using Tavily search and synthesizes
        the results into a structured research report.

    Inputs (AgentState):
        - topic (str): The topic to research

    Outputs (AgentState updates):
        - research_data (str): Structured research summary
        - sources (list[dict]): List of tracked sources
        - error (str, optional): Error message if failure occurs
    """
    try:
        topic = state.get("topic", "").strip()
        if not topic:
            return {"error": "Topic cannot be empty"}

        print("--- RESEARCHER AGENT WORKING ---")
        print(f"Researching topic: {topic}")

        search_results = tavily_tool.invoke(topic)
        if not search_results:
            return {"error": "No search results found for the given topic"}

        sources = []
        search_content = ""

        for result in search_results:
            sources.append({
                "url": result.get("url"),
                "title": result.get("title"),
                "source": result.get("source"),
            })

            search_content += (
                f"Source: {result.get('url')}\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"Content: {result.get('content')}\n\n"
            )

        chain = RESEARCHER_PROMPT | llm
        response = chain.invoke({
            "topic": topic,
            "search_content": search_content,
        })

        return {
            "research_data": response.content,
            "sources": sources,
        }

    except LangChainException as e:
        return {"error": f"Researcher node failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected researcher error: {str(e)}"}


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

        chain = WRITER_PROMPT | llm_creative
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
