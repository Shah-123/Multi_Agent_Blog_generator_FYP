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
    GlobalImagePlan
)
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
from Graph.keyword_optimizer import keyword_optimizer_node

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
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        return out
    except Exception as e:
        print(f"⚠️ Search failed for '{query}': {e}")
        return []

def _safe_slug(title: str) -> str:
    """Creates a filename-safe slug from a string."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"

# ------------------------------------------------------------------
# 1. ROUTER NODE
# ------------------------------------------------------------------
def router_node(state: State) -> dict:
    print("--- 🚦 ROUTING ---")
    decider = llm.with_structured_output(RouterDecision)
    as_of = state.get("as_of", date.today().isoformat())
    
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {as_of}"),
    ])

    # Determine context window
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

    # Deduplicate by URL
    dedup = {e.url: e for e in pack.evidence if e.url}
    evidence = list(dedup.values())
    
    print(f"   ✅ Found {len(evidence)} evidence items.")
    return {"evidence": evidence}

# ------------------------------------------------------------------
# 3. ORCHESTRATOR NODE (PLANNER) - UPDATED WITH TONE & KEYWORDS
# ------------------------------------------------------------------
def orchestrator_node(state: State) -> dict:
    print("--- 📋 PLANNING ---")
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    
    # NEW: Get tone and keywords from state
    target_tone = state.get("target_tone", "professional")
    target_keywords = state.get("target_keywords", [])
    
    # Format keywords for prompt
    keywords_str = ", ".join(target_keywords) if target_keywords else "None specified"
    
    # Build prompt with tone and keywords
    prompt_content = f"""Topic: {state['topic']}
Mode: {mode}
Target Tone: {target_tone}
Target Keywords: {keywords_str}

Evidence Context:
{[e.model_dump() for e in evidence][:10]}

Create a blog plan that:
1. Maintains '{target_tone}' tone consistently throughout all sections
2. Naturally integrates these keywords: {keywords_str}
3. Distributes keywords strategically across sections (avoid stuffing)
4. Creates engaging, SEO-optimized content
"""
    
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM.format(
            tone=target_tone, 
            keywords=keywords_str
        )),
        HumanMessage(content=prompt_content),
    ])
    
    print(f"   Generated {len(plan.tasks)} sections.")
    print(f"   🎨 Tone: {plan.tone}")
    if plan.primary_keywords:
        print(f"   🎯 Keywords: {', '.join(plan.primary_keywords)}")
    
    return {"plan": plan}

# ------------------------------------------------------------------
# 4. FANOUT (PARALLEL DISPATCHER)
# ------------------------------------------------------------------
def fanout(state: State):
    """Generates parallel workers for each section."""
    if not state.get("plan"):
        print("⚠️ No plan found, skipping fanout.")
        return []
        
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

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    
    print(f"   ✍️ Writing Section {task.id + 1}/{len(plan.tasks)}: {task.title} (Tone: {plan.tone})")

    try:
        bullets_text = "\n- " + "\n- ".join(task.bullets)
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'Unknown Date'}"
            for e in evidence[:15]
        )
        
        section_keywords = task.tags[:3]
        keywords_str = ", ".join(section_keywords) if section_keywords else "general topic"

        # INCREASED: max_tokens from default to 3000 for longer sections
        response = llm.invoke(
            [
                SystemMessage(content=WORKER_SYSTEM.format(
                    tone=plan.tone,
                    keywords=keywords_str,
                    target_words=task.target_words
                )),
                HumanMessage(content=(
                    f"Blog Title: {plan.blog_title}\n"
                    f"Section Number: {task.id + 1} of {len(plan.tasks)}\n"
                    f"Section Title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target Words: {task.target_words}\n"
                    f"Tone: {plan.tone} (MAINTAIN THIS TONE CONSISTENTLY)\n"
                    f"Keywords to integrate naturally: {keywords_str}\n"
                    f"Bullets to Cover:{bullets_text}\n\n"
                    f"Available Evidence (Cite these URLs):\n{evidence_text}\n\n"
                    f"CRITICAL INSTRUCTIONS:\n"
                    f"1. Write EXACTLY {task.target_words} words (minimum)\n"
                    f"2. Cover ALL bullet points completely\n"
                    f"3. End with a complete sentence (period/question mark/exclamation)\n"
                    f"4. DO NOT stop mid-sentence or mid-paragraph\n\n"
                    f"Remember: Write in {plan.tone} tone throughout."
                )),
            ],
            max_tokens=3000  # INCREASED from default (~1000)
        )
        
        section_md = response.content.strip()
        
        # Validation
        word_count = len(section_md.split())
        if word_count < (task.target_words * 0.7):
            print(f"   ⚠️ Section {task.id + 1} seems short ({word_count} words, target: {task.target_words})")
        
        if not section_md.endswith(('.', '!', '?', '"', ')')):
            print(f"   ⚠️ Section {task.id + 1} incomplete (doesn't end with punctuation)")
            section_md += "."
        
        print(f"   ✅ Completed: {word_count} words")
        
    except Exception as e:
        import traceback
        print(f"   ❌ Error in section {task.title}: {e}")
        traceback.print_exc()
        section_md = f"## {task.title}\n\n[Error generating content: {str(e)}]"

    return {"sections": [(task.id, section_md)]}
# . REDUCER: MERGE CONTENT (FIXED: Deduplicates sections)
# ------------------------------------------------------------------
def merge_content(state: State) -> dict:
    print("--- 🔗 MERGING SECTIONS ---")
    plan = state["plan"]
    
    # DEDUPLICATION LOGIC
    # Workers might append duplicates if the graph is resumed.
    # We use a dictionary to keep only the LATEST content for each Task ID.
    unique_sections = {}
    for task_id, content in state["sections"]:
        unique_sections[task_id] = content
        
    # Sort by Task ID to ensure correct order
    ordered_content = [unique_sections[k] for k in sorted(unique_sections.keys())]
    
    body = "\n\n".join(ordered_content).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    
    print(f"   ✅ Merged {len(ordered_content)} sections")
    
    return {"merged_md": merged_md}

# ------------------------------------------------------------------
# 7. REDUCER: DECIDE IMAGES
# ------------------------------------------------------------------
def decide_images(state: State) -> dict:
    print("--- 🖼️ PLANNING IMAGES ---")
    planner = llm.with_structured_output(GlobalImagePlan)
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Current Blog Content:\n{state['merged_md']}" 
        )),
    ])

    return {
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

# ------------------------------------------------------------------
# 8. REDUCER: GENERATE IMAGES & SAVE
# ------------------------------------------------------------------
def _generate_image_bytes_google(prompt: str) -> Optional[bytes]:
    """Generates image using Google GenAI (Gemini)."""
    try:
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
    except ImportError:
        print("   ⚠️ Google GenAI library not installed.")
        return None
    except Exception as e:
        print(f"   ⚠️ Image Gen Error: {e}")
        return None

def generate_and_place_images(state: State) -> dict:
    """Generates images, replaces placeholders, and returns final text."""
    print("--- 🎨 GENERATING IMAGES & SAVING ---")
    
    plan = state["plan"]
    final_md = state.get("merged_md", "")
    image_specs = state.get("image_specs", [])
    
    # Use the folder path passed in state, or default to current dir
    base_path = state.get("blog_folder", ".")
    assets_path = f"{base_path}/assets/images"
    
    # 1. Try to generate images
    if os.getenv("GOOGLE_API_KEY") and image_specs:
        print(f"   Attempting to generate {len(image_specs)} images...")
        
        # Create images dir if not exists
        Path(assets_path).mkdir(parents=True, exist_ok=True)

        for img in image_specs:
            img_bytes = _generate_image_bytes_google(img["prompt"])
            
            if img_bytes:
                img_filename = _safe_slug(img["filename"])
                if not img_filename.endswith(".png"): img_filename += ".png"
                
                full_path = Path(f"{assets_path}/{img_filename}")
                full_path.write_bytes(img_bytes)
                
                # Use relative path for Markdown compatibility
                rel_path = f"../assets/images/{img_filename}"
                markdown_image = f"\n\n![{img['alt']}]({rel_path})\n*Figure: {img['caption']}*\n\n"
                
                # Find the target paragraph and inject the image AFTER IT
                target_phrase = img.get("target_paragraph", "")
                if target_phrase:
                    # Escape regex characters just in case
                    import re
                    # Look for the target phrase, then match until the end of that paragraph (double newline)
                    # We use a regex that finds the phrase, then any characters up to the next \n\n or end of string
                    escaped_phrase = re.escape(target_phrase)
                    # Pattern: escaped_phrase + anything (non-greedy) + (\n\n or end of string)
                    pattern = re.compile(rf"({escaped_phrase}.*?(?:\n\n|\Z))", re.DOTALL)
                    
                    # Check if it matches
                    if pattern.search(final_md):
                        # Replace the first occurrence: append the image right after the matched paragraph
                        final_md = pattern.sub(rf"\1{markdown_image}", final_md, count=1)
                    else:
                        print(f"   ⚠️ Could not find target paragraph starting with '{target_phrase}', appending to end.")
                        final_md += markdown_image
                else:
                    final_md += markdown_image

                print(f"   ✅ Generated: {img_filename}")
            else:
                print(f"   ❌ Failed: {img['filename']} (skipping)")
                
    else:
        print("   ⏭️ Skipped Image Generation (No API Key or no specs)")

    return {"final": final_md}

# ------------------------------------------------------------------
# 9. FACT CHECKER NODE
# ------------------------------------------------------------------
def fact_checker_node(state: State) -> dict:
    print("--- 🕵️ FACT CHECKING ---")
    checker = llm.with_structured_output(FactCheckReport)
    
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
    
    report_text = f"FACT CHECK REPORT\n"
    report_text += "=" * 60 + "\n"
    report_text += f"Score: {report.score}/10\n"
    report_text += f"Verdict: {report.verdict}\n\n"
    
    if report.issues:
        report_text += f"Issues Found: {len(report.issues)}\n\n"
        report_text += "DETAILS:\n"
        for i, issue in enumerate(report.issues, 1):
            report_text += f"{i}. [{issue.issue_type}] {issue.claim}\n"
            report_text += f"   -> Fix: {issue.recommendation}\n\n"
    else:
        report_text += "✅ No issues found!\n"
    
    print(f"   📊 Score: {report.score}/10 | Verdict: {report.verdict}")
    return {"fact_check_report": report_text}

# ------------------------------------------------------------------
# 10. SOCIAL MEDIA NODE
# ------------------------------------------------------------------
def social_media_node(state: State) -> dict:
    print("--- 📱 GENERATING SOCIAL MEDIA PACK ---")
    
    blog_post = state["final"]
    topic = state["topic"]
    evidence = state.get("evidence", [])
    
    key_stats = "\n".join([f"- {e.snippet[:100]}... ({e.url})" for e in evidence[:5]])
    
    # Construct Context Once
    context = f"TOPIC: {topic}\nBLOG CONTENT:\n{blog_post[:4000]}\nSTATS:\n{key_stats}"
    
    # Parallelize calls ideally, but sequential is fine for now
    linkedin = llm.invoke([SystemMessage(content=LINKEDIN_SYSTEM), HumanMessage(content=context)]).content
    print("   ✅ LinkedIn Generated")
    
    youtube = llm.invoke([SystemMessage(content=YOUTUBE_SYSTEM), HumanMessage(content=context)]).content
    print("   ✅ YouTube Generated")
    
    facebook = llm.invoke([SystemMessage(content=FACEBOOK_SYSTEM), HumanMessage(content=context)]).content
    print("   ✅ Facebook Generated")
    
    return {
        "linkedin_post": linkedin,
        "youtube_script": youtube,
        "facebook_post": facebook
    }

# ------------------------------------------------------------------
# 11. EVALUATOR NODE
# ------------------------------------------------------------------
def evaluator_node(state: State) -> dict:
    print("--- 📊 EVALUATING QUALITY ---")
    try:
        from validators import BlogEvaluator
        evaluator = BlogEvaluator()
        results = evaluator.evaluate(blog_post=state["final"], topic=state["topic"])
        print(f"   🏆 Final Score: {results.get('final_score')}/10")
        return {"quality_evaluation": results}
    except ImportError:
        print("   ⚠️ Validators module not found, skipping evaluation.")
        return {"quality_evaluation": {"error": "Module missing"}}
    except Exception as e:
        print(f"   ⚠️ Evaluation Error: {e}")
        return {"quality_evaluation": {"error": str(e)}}