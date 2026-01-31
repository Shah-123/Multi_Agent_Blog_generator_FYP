import os
from datetime import date, timedelta
from typing import List, Optional
from pathlib import Path
import re

from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# ------------------------------------------------------------------
# IMPORT SCHEMAS & TEMPLATES
# ------------------------------------------------------------------
from Graph.state import (
    State, 
    RouterDecision, 
    EvidencePack, 
    Plan, 
    Task, 
    EvidenceItem, 
    GlobalImagePlan,
    ImageSpec
)
from Graph.templates import (
    ROUTER_SYSTEM, 
    RESEARCH_SYSTEM, 
    ORCH_SYSTEM, 
    WORKER_SYSTEM, 
    DECIDE_IMAGES_SYSTEM
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Safe Tavily search wrapper."""
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️ TAVILY_API_KEY missing. Skipping search.")
        return []
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append({
                "title": r.get("title") or "",
                "url": r.get("url") or "",
                "snippet": r.get("content") or r.get("snippet") or "",
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        return out
    except Exception as e:
        print(f"⚠️ Search failed for '{query}': {e}")
        return []

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s: return None
    try: return date.fromisoformat(s[:10])
    except: return None

def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

# ------------------------------------------------------------------
# 1. ROUTER NODE
# ------------------------------------------------------------------
def router_node(state: State) -> dict:
    """Decides if we need research and what mode to run in."""
    print("--- 🚦 ROUTING ---")
    decider = llm.with_structured_output(RouterDecision)
    
    # Ensure as_of is set
    as_of = state.get("as_of", date.today().isoformat())
    
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
    ])

    # Determine context window (recency)
    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650 # 10 years (effectively forever)

    print(f"   Mode: {decision.mode} | Research Needed: {decision.needs_research}")
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "as_of": as_of
    }

# ------------------------------------------------------------------
# 2. RESEARCH NODE
# ------------------------------------------------------------------
def research_node(state: State) -> dict:
    """Performs web search and extracts structured evidence."""
    print("--- 🔍 RESEARCHING ---")
    
    queries = (state.get("queries") or [])[:5] # Limit to 5 queries to save tokens
    raw_results: List[dict] = []
    
    for q in queries:
        print(f"   Searching: {q}")
        raw_results.extend(_tavily_search(q, max_results=4))

    if not raw_results:
        print("   ⚠️ No results found.")
        return {"evidence": []}

    print("   📊 Extracting Evidence...")
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state['as_of']}\n"
            f"Recency days: {state['recency_days']}\n\n"
            f"Raw results:\n{str(raw_results)[:15000]}" # Truncate to avoid context errors
        )),
    ])

    # Deduplicate by URL
    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())
    
    print(f"   ✅ Found {len(evidence)} evidence items.")
    return {"evidence": evidence}

# ------------------------------------------------------------------
# 3. ORCHESTRATOR NODE (PLANNER)
# ------------------------------------------------------------------
def orchestrator_node(state: State) -> dict:
    """Generates the blog plan/outline."""
    print("--- 📋 PLANNING ---")
    
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"Evidence Context:\n{[e.model_dump() for e in evidence][:10]}"
        )),
    ])
    
    print(f"   Generated {len(plan.tasks)} sections.")
    return {"plan": plan}

# ------------------------------------------------------------------
# 4. FANOUT (PARALLEL DISPATCHER)
# ------------------------------------------------------------------
def fanout(state: State):
    """Generates parallel workers for each section."""
    if not state["plan"]:
        raise ValueError("No plan found in state!")
        
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]

# ------------------------------------------------------------------
# 5. WORKER NODE (WRITER)
# ------------------------------------------------------------------
def worker_node(payload: dict) -> dict:
    """Writes a single section of the blog."""
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    
    print(f"   ✍️ Writing Section: {task.title}")

    # Format inputs for the LLM
    bullets_text = "\n- " + "\n- ".join(task.bullets)
    evidence_text = "\n".join(
        f"- {e.title} | {e.url} | {e.published_at or 'Unknown Date'}"
        for e in evidence[:15]
    )

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog Title: {plan.blog_title}\n"
            f"Section: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target Words: {task.target_words}\n"
            f"Bullets to Cover:{bullets_text}\n\n"
            f"Available Evidence (Cite these URLs):\n{evidence_text}\n"
        )),
    ]).content.strip()

    # Return as a tuple (id, content) for re-sorting later
    return {"sections": [(task.id, section_md)]}

# ------------------------------------------------------------------
# 6. REDUCER: MERGE CONTENT
# ------------------------------------------------------------------
def merge_content(state: State) -> dict:
    """Combines all written sections into one markdown document."""
    print("--- 🔗 MERGING SECTIONS ---")
    plan = state["plan"]
    
    # Sort by Task ID to ensure correct order
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    
    return {"merged_md": merged_md}

# ------------------------------------------------------------------
# 7. REDUCER: DECIDE IMAGES
# ------------------------------------------------------------------
def decide_images(state: State) -> dict:
    """Decides where to place images in the merged text."""
    print("--- 🖼️ PLANNING IMAGES ---")
    planner = llm.with_structured_output(GlobalImagePlan)
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Current Blog Content:\n{state['merged_md'][:10000]}..." # Truncate check
        )),
    ])

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

# ------------------------------------------------------------------
# 8. REDUCER: GENERATE IMAGES (SAFE VERSION)
# ------------------------------------------------------------------
def _generate_image_bytes_google(prompt: str) -> Optional[bytes]:
    """Generates image using Google GenAI (Gemini)."""
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return None

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp", # Updated model name
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        
        # Extract bytes (New SDK format)
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
        return None
    except Exception as e:
        print(f"   ⚠️ Image Generation Failed: {e}")
        return None

def generate_and_place_images(state: State) -> dict:
    """Generates images and replaces placeholders."""
    print("--- 🎨 GENERATING IMAGES ---")
    
    plan = state["plan"]
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Prepare output directory
    images_dir = Path("generated_images")
    images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        
        print(f"   🎨 Generating: {spec['prompt'][:50]}...")
        
        # Try generation
        img_bytes = _generate_image_bytes_google(spec["prompt"])
        
        if img_bytes:
            # Save Image
            out_path = images_dir / filename
            out_path.write_bytes(img_bytes)
            # Link in Markdown
            img_md = f"![{spec['alt']}](generated_images/{filename})\n*{spec['caption']}*"
            md = md.replace(placeholder, img_md)
        else:
            # Fallback text if generation fails/skipped
            fallback = f"> *[Image Suggested: {spec['caption']}]*"
            md = md.replace(placeholder, fallback)

    # Save Final File
    final_filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(final_filename).write_text(md, encoding="utf-8")
    
    print(f"   ✅ Saved final blog to: {final_filename}")
    return {"final": md}