import os
import re
from datetime import date
from typing import List, Optional
from pathlib import Path

# LangChain / LangGraph Imports
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# Internal Imports
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
# Note: Ensure you deleted the duplicate definitions in templates.py as advised!
from Graph.templates import (
    ROUTER_SYSTEM, 
    RESEARCH_SYSTEM, 
    ORCH_SYSTEM, 
    WORKER_SYSTEM, 
    DECIDE_IMAGES_SYSTEM,
    LINKEDIN_SYSTEM,
    YOUTUBE_SYSTEM,
    FACEBOOK_SYSTEM,
    FACT_CHECKER_SYSTEM
)
from Graph.structured_data import FactCheckReport

# Initialize LLM
# optimize: model="gpt-4o" is better for the Orchestrator, but "gpt-4o-mini" is cheaper
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

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
        recency_days = 3650 # 10 years

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
    
    queries = (state.get("queries") or [])[:5] 
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
            f"Raw results:\n{str(raw_results)}" 
        )),
    ])

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

    try:
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
                f"Tone: {plan.tone}\n"  # Added tone from plan
                f"Bullets to Cover:{bullets_text}\n\n"
                f"Available Evidence (Cite these URLs):\n{evidence_text}\n"
            )),
        ]).content.strip()
    except Exception as e:
        print(f"   ❌ Error in section {task.title}: {e}")
        section_md = f"## {task.title}\n\n[Error generating content: {str(e)}]"

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
            f"Current Blog Content:\n{state['merged_md'][:10000]}..." 
        )),
    ])

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

# ------------------------------------------------------------------
# 8. REDUCER: GENERATE IMAGES & SAVE
# ------------------------------------------------------------------
def _generate_image_bytes_google(prompt: str) -> Optional[bytes]:
    """Generates image using Google GenAI (Gemini)."""
    try:
        # Lazy import to avoid crash if library missing
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: 
            return None

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image", 
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
        return None
    except Exception as e:
        print(f"   ⚠️ Image Gen Error: {e}")
        return None

def generate_and_place_images(state: State) -> dict:
    """
    Generates images if API key exists, otherwise saves text-only.
    Also acts as the final 'Saver' node.
    """
    print("--- 🎨 GENERATING IMAGES & SAVING ---")
    
    plan = state["plan"]
    final_md = state.get("md_with_placeholders", state["merged_md"])
    image_specs = state.get("image_specs", [])
    
    # 1. Try to generate images
    if os.getenv("GOOGLE_API_KEY") and image_specs:
        print(f"   Attempting to generate {len(image_specs)} images...")
        saved_images = []
        
        for img in image_specs:
            img_bytes = _generate_image_bytes_google(img["prompt"])
            if img_bytes:
                # Save image file
                img_filename = _safe_slug(img["filename"])
                # Ensure extension
                if not img_filename.endswith(".png"): img_filename += ".png"
                
                # Create images dir if not exists
                Path("generated_images").mkdir(exist_ok=True)
                path = Path(f"generated_images/{img_filename}")
                path.write_bytes(img_bytes)
                
                # Replace placeholder in Markdown with actual image link
                # Format: ![Alt Text](path/to/image.png)
                rel_path = f"./generated_images/{img_filename}"
                final_md = final_md.replace(
                    img["placeholder"], 
                    f"![{img['alt']}]({rel_path})\n*Figure: {img['caption']}*\n"
                )
                print(f"   ✅ Generated: {img_filename}")
            else:
                print(f"   ❌ Failed to generate: {img['filename']}")
                # Remove placeholder
                final_md = final_md.replace(img["placeholder"], "")
    else:
        print("   ⏭️ Skipped Image Generation (No API Key or no images planned)")
        # Clean up placeholders if we skipped generation
        final_md = re.sub(r"\[\[IMAGE_\d+\]\]", "", final_md)

    # 2. Save Final Markdown
    final_filename = f"{_safe_slug(plan.blog_title)}.md"
    Path(final_filename).write_text(final_md, encoding="utf-8")
    
    print(f"   ✅ Saved final blog to: {final_filename}")
    
    # Return 'final' key which subsequent nodes (FactCheck/Social) use
    return {"final": final_md}

# ------------------------------------------------------------------
# 9. FACT CHECKER NODE
# ------------------------------------------------------------------
def fact_checker_node(state: State) -> dict:
    """Verifies the final blog content using structured template."""
    print("--- 🕵️ FACT CHECKING ---")
    
    # Use structured output for consistent reporting
    checker = llm.with_structured_output(FactCheckReport)
    
    # Prepare evidence context
    evidence_summary = "\n".join([
        f"- {e.title} | {e.url}"
        for e in state.get("evidence", [])[:20]
    ])
    
    report = checker.invoke([
        SystemMessage(content=FACT_CHECKER_SYSTEM),
        HumanMessage(content=(
            f"BLOG CONTENT TO AUDIT:\n{state['final'][:8000]}\n\n"
            f"EVIDENCE USED IN RESEARCH:\n{evidence_summary}"
        ))
    ])
    
    # Format for display/storage
    report_text = f"""
FACT CHECK REPORT
═════════════════
Score: {report.score}/10
Verdict: {report.verdict}

Issues Found: {len(report.issues)}
"""
    if report.issues:
        report_text += "\nDETAILS:\n"
        for i, issue in enumerate(report.issues, 1):
            report_text += f"{i}. [{issue.issue_type}] {issue.claim}\n   -> Fix: {issue.recommendation}\n"
    
    print(f"   📊 Score: {report.score}/10 | Verdict: {report.verdict}")
    return {"fact_check_report": report_text}

# ------------------------------------------------------------------
# 10. SOCIAL MEDIA NODE (Professional Version)
# ------------------------------------------------------------------
def social_media_node(state: State) -> dict:
    """Generates LinkedIn, YouTube, and Facebook content using professional templates."""
    print("--- 📱 GENERATING SOCIAL MEDIA PACK ---")
    
    blog_post = state["final"]
    topic = state["topic"]
    
    # Extract key stats for context
    evidence = state.get("evidence", [])
    key_stats = "\n".join([
        f"- {e.snippet[:100]}... ({e.url})"
        for e in evidence[:5]
    ])
    
    context = f"""
ORIGINAL BLOG TOPIC: {topic}
BLOG CONTENT (excerpt):
{blog_post[:5000]}

KEY STATISTICS FROM RESEARCH:
{key_stats}
"""
    
    # 1. LinkedIn
    print("   📝 Generating LinkedIn post...")
    linkedin = llm.invoke([
        SystemMessage(content=LINKEDIN_SYSTEM),
        HumanMessage(content=context)
    ]).content
    
    # 2. YouTube
    print("   🎬 Generating YouTube script...")
    youtube = llm.invoke([
        SystemMessage(content=YOUTUBE_SYSTEM),
        HumanMessage(content=context)
    ]).content
    
    # 3. Facebook
    print("   👥 Generating Facebook post...")
    facebook = llm.invoke([
        SystemMessage(content=FACEBOOK_SYSTEM),
        HumanMessage(content=context)
    ]).content
    
    print("   ✅ Social Media Content Generated")
    
    return {
        "linkedin_post": linkedin,
        "youtube_script": youtube,
        "facebook_post": facebook
    }

# ------------------------------------------------------------------
# 11. EVALUATOR NODE
# ------------------------------------------------------------------
def evaluator_node(state: State) -> dict:
    """Scores the blog using your validator logic."""
    print("--- 📊 EVALUATING QUALITY ---")
    
    try:
        from validators import BlogEvaluator
        evaluator = BlogEvaluator()
        results = evaluator.evaluate(
            blog_post=state["final"],
            topic=state["topic"]
        )
        print(f"   🏆 Final Score: {results.get('final_score')}/10")
        return {"quality_evaluation": results}
        
    except ImportError:
        print("   ⚠️ Validators module not found, skipping evaluation.")
        return {"quality_evaluation": {"error": "Module missing"}}