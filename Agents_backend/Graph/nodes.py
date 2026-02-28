import os
import re
import logging
from datetime import date
from typing import List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# LOGGING SETUP
# ------------------------------------------------------------------
logger = logging.getLogger("blog_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if os.getenv("DEBUG") else logging.INFO)

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
    FACT_CHECKER_SYSTEM,
    REVISION_SYSTEM
)
from Graph.structured_data import FactCheckReport
from Graph.keyword_optimizer import keyword_optimizer_node
from event_bus import emit as _emit

def _job(state) -> str:
    """Extract job ID from state (works for both State dict and payload dict)."""
    return state.get("_job_id", "")

# ------------------------------------------------------------------
# LLM MODEL TIERS (configurable via environment variables)
# ------------------------------------------------------------------
# Fast model: routing, research extraction, social media, image planning
# Quality model: content writing, fact-checking, revision
_FAST_MODEL = os.getenv("LLM_FAST_MODEL", "gpt-4o-mini")
_QUALITY_MODEL = os.getenv("LLM_QUALITY_MODEL", "gpt-4o")

llm_fast = ChatOpenAI(model=_FAST_MODEL, temperature=0)
llm_quality = ChatOpenAI(model=_QUALITY_MODEL, temperature=0.1)

# Backward compat alias — will remove eventually
llm = llm_fast

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Safe Tavily search wrapper."""
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY missing. Skipping search.")
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
        logger.warning(f"Search failed for '{query}': {e}")
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
    _emit(_job(state), "router", "started", "Analyzing topic and deciding research strategy...")
    logger.info("🚦 ROUTING ---")
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

    logger.info(f"Mode: {decision.mode} | Research Needed: {decision.needs_research}")
    _emit(_job(state), "router", "completed", f"Mode: {decision.mode} | Research: {decision.needs_research}", {"mode": decision.mode, "needs_research": decision.needs_research})
    
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
        "as_of": as_of
    }



def scrape_full_webpage(url: str, max_words: int = 1500) -> str:
    """Visits a URL and scrapes the actual article text."""
    try:
        # We use a standard browser User-Agent so websites don't block us
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove junk like scripts, styles, and footers
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()

        # Get the clean text
        text = soup.get_text(separator=' ', strip=True)
        
        # Limit the text size so we don't blow up the LLM token limit
        words = text.split()
        return " ".join(words[:max_words])
    
    except Exception as e:
        logger.warning(f"Failed to scrape {url}: {e}")
        return ""

# ------------------------------------------------------------------
# 2. RESEARCH NODE
# ------------------------------------------------------------------
def research_node(state: State) -> dict:
    _emit(_job(state), "research", "started", "Searching the web for evidence...")
    logger.info("🔍 DEEP RESEARCHING ---")
    queries = (state.get("queries") or [])[:3] # Keep it to 3 searches to save time
    
    # 1. Gather URLs from Tavily
    found_urls = set()
    raw_results = []
    
    for q in queries:
        logger.info(f"Searching: {q}")
        _emit(_job(state), "research", "working", f"Searching: {q}")
        results = _tavily_search(q, max_results=3)
        for r in results:
            if r['url'] not in found_urls:
                found_urls.add(r['url'])
                raw_results.append(r)

    if not raw_results:
        logger.warning("No results found.")
        _emit(_job(state), "research", "completed", "No results found", {"sources": 0})
        return {"evidence": []}

    logger.info(f"🕸️ Scraping {len(raw_results[:5])} top articles...")
    _emit(_job(state), "research", "working", f"Deep-scraping {len(raw_results[:5])} articles...")
    
    # 2. Scrape the full text of the top 5 URLs
    deep_evidence_context = ""
    for idx, r in enumerate(raw_results[:5]):
        logger.info(f"-> Reading: {r['url']}")
        full_text = scrape_full_webpage(r['url'])
        
        if full_text:
            deep_evidence_context += f"SOURCE {idx+1}: {r['title']} ({r['url']})\n"
            deep_evidence_context += f"CONTENT: {full_text[:3000]}\n\n" # Send up to 3000 chars per site

    logger.info("🧠 Analyzing full articles for hard evidence...")
    _emit(_job(state), "research", "working", "Extracting verified facts from articles...")
    
    # 3. Give ALL the scraped text to the LLM to extract the best facts
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Read the following full articles and extract ONLY hard facts, statistics, and verifiable claims.\n\n"
            f"SCRAPED ARTICLES:\n{deep_evidence_context}" 
        )),
    ])

    evidence = list({e.url: e for e in pack.evidence if e.url}.values()) # Deduplicate
    
    logger.info(f"✅ Extracted {len(evidence)} verified deep-evidence items.")
    _emit(_job(state), "research", "completed", f"Found {len(evidence)} verified evidence items", {"sources": len(evidence)})
    return {"evidence": evidence}


# ------------------------------------------------------------------
# 3. ORCHESTRATOR NODE (PLANNER) - UPDATED WITH TONE & KEYWORDS
# ------------------------------------------------------------------
def orchestrator_node(state: State) -> dict:
    _emit(_job(state), "orchestrator", "started", "Creating detailed blog outline...")
    logger.info("📋 PLANNING ---")
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
    
    logger.info(f"Generated {len(plan.tasks)} sections.")
    logger.info(f"🎨 Tone: {plan.tone}")
    if plan.primary_keywords:
        logger.info(f"🎯 Keywords: {', '.join(plan.primary_keywords)}")
    
    _emit(_job(state), "orchestrator", "completed", f"Planned {len(plan.tasks)} sections", {"sections": len(plan.tasks), "tone": plan.tone})
    return {"plan": plan}

# ------------------------------------------------------------------
# 4. FANOUT (PARALLEL DISPATCHER)
# ------------------------------------------------------------------
def fanout(state: State):
    """Generates parallel workers for each section."""
    if not state.get("plan"):
        logger.warning("No plan found, skipping fanout.")
        return []
    
    _emit(_job(state), "writer", "started", f"Dispatching {len(state['plan'].tasks)} parallel writers...")
    
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "mode": state["mode"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
            "_job_id": state.get("_job_id", ""),
        })
        for task in state["plan"].tasks
    ]

def worker_node(payload: dict) -> dict:
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    job_id = payload.get("_job_id", "")
    
    _emit(job_id, "writer", "working", f"Writing section {task.id + 1}/{len(plan.tasks)}: {task.title}", {"section": task.id + 1, "total": len(plan.tasks)})
    logger.info(f"✍️ Writing Section {task.id + 1}/{len(plan.tasks)}: {task.title} (Tone: {plan.tone})")

    try:
        bullets_text = "\n- " + "\n- ".join(task.bullets)
        evidence_text = "\n".join(
            f"- [{e.title}]({e.url}) ({e.published_at or 'Unknown Date'})\n  Content: {e.snippet[:300]}"
            for e in evidence[:15]
        )
        
        section_keywords = task.tags[:3]
        keywords_str = ", ".join(section_keywords) if section_keywords else "general topic"

        # INCREASED: max_tokens from default to 3000 for longer sections
        response = llm_quality.invoke(
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
        
        # Normalize heading: strip any heading the LLM may have added, then prepend H2
        lines = section_md.split('\n')
        if lines and re.match(r'^#{1,4}\s+', lines[0]):
            # LLM added its own heading — remove it, we'll add the correct one
            lines = lines[1:]
            section_md = '\n'.join(lines).strip()
        section_md = f"## {task.title}\n\n{section_md}"
        
        # Validation
        word_count = len(section_md.split())
        if word_count < (task.target_words * 0.7):
            logger.warning(f"Section {task.id + 1} seems short ({word_count} words, target: {task.target_words})")
        
        if not section_md.endswith(('.', '!', '?', '"', ')')):
            logger.warning(f"Section {task.id + 1} incomplete (doesn't end with punctuation)")
            section_md += "."
        
        logger.info(f"✅ Completed: {word_count} words")
        _emit(job_id, "writer", "working", f"Completed section {task.id + 1}: {task.title} ({word_count} words)", {"section": task.id + 1, "words": word_count})
        
    except Exception as e:
        import traceback
        logger.error(f"Error in section {task.title}: {e}")
        traceback.print_exc()
        section_md = f"## {task.title}\n\n[Error generating content: {str(e)}]"
        _emit(job_id, "writer", "error", f"Failed section {task.id + 1}: {str(e)}")

    return {"sections": [(task.id, section_md)]}
# . REDUCER: MERGE CONTENT (FIXED: Deduplicates sections)
# ------------------------------------------------------------------
def merge_content(state: State) -> dict:
    _emit(_job(state), "merger", "started", "Merging all sections into final blog...")
    logger.info("🔗 MERGING SECTIONS ---")
    plan = state["plan"]
    
    # DEDUPLICATION LOGIC
    unique_sections = {}
    for task_id, content in state["sections"]:
        unique_sections[task_id] = content
        
    ordered_content = [unique_sections[k] for k in sorted(unique_sections.keys())]
    
    body = "\n\n".join(ordered_content).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    
    word_count = len(merged_md.split())
    logger.info(f"✅ Merged {len(ordered_content)} sections")
    _emit(_job(state), "merger", "completed", f"Merged {len(ordered_content)} sections ({word_count} words)", {"sections": len(ordered_content), "words": word_count})
    _emit(_job(state), "writer", "completed", f"All {len(ordered_content)} sections written", {"sections": len(ordered_content), "words": word_count})
    
    return {"merged_md": merged_md}

# ------------------------------------------------------------------
# 7. REDUCER: DECIDE IMAGES
# ------------------------------------------------------------------
def decide_images(state: State) -> dict:
    _emit(_job(state), "images", "started", "Planning image placement...")
    logger.info("🖼️ PLANNING IMAGES ---")
    planner = llm.with_structured_output(GlobalImagePlan)
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Current Blog Content:\n{state['merged_md']}" 
        )),
    ])

    _emit(_job(state), "images", "working", f"Planned {len(image_plan.images)} images", {"count": len(image_plan.images)})
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
        logger.warning("Google GenAI library not installed.")
        return None
    except Exception as e:
        logger.warning(f"Image Gen Error: {e}")
        return None

def generate_and_place_images(state: State) -> dict:
    """Generates images, replaces placeholders, and returns final text."""
    _emit(_job(state), "images", "working", "Generating AI images...")
    logger.info("🎨 GENERATING IMAGES & SAVING ---")
    
    plan = state["plan"]
    final_md = state.get("merged_md", "")
    image_specs = state.get("image_specs", [])
    
    # Use the folder path passed in state, or default to current dir
    base_path = state.get("blog_folder", ".")
    assets_path = f"{base_path}/assets/images"
    
    # 1. Try to generate images
    if os.getenv("GOOGLE_API_KEY") and image_specs:
        logger.info(f"Attempting to generate {len(image_specs)} images...")
        
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
                target_phrase = img.get("target_paragraph", "").strip()
                if target_phrase:
                    # Break the generated blog into paragraphs
                    paragraphs = re.split(r'\n\s*\n', final_md)
                    target_words = set(target_phrase.lower().split())
                    
                    best_match_idx = -1
                    best_score = 0
                    
                    for i, p in enumerate(paragraphs):
                        p_words = set(p.lower().split())
                        # Calculate Jaccard-like overlap: how many target words are in this paragraph?
                        overlap = len(target_words.intersection(p_words))
                        if overlap > best_score:
                            best_score = overlap
                            best_match_idx = i
                    
                    # If we found a reasonable match (e.g. at least 3 matching words, or 30% of target)
                    if best_match_idx >= 0 and best_score >= min(3, len(target_words) // 3):
                        # Insert the image after the matched paragraph
                        paragraphs.insert(best_match_idx + 1, markdown_image.strip())
                        final_md = "\n\n".join(paragraphs)
                    else:
                        logger.warning(f"Could not find target paragraph similar to '{target_phrase}', appending to end.")
                        final_md += "\n" + markdown_image
                else:
                    final_md += "\n" + markdown_image

                logger.info(f"✅ Generated: {img_filename}")
            else:
                logger.error(f"Failed: {img['filename']} (skipping)")
                
    else:
        logger.info("⏭️ Skipped Image Generation (No API Key or no specs)")

    _emit(_job(state), "images", "completed", "Images processed")
    return {"final": final_md}

# ------------------------------------------------------------------
# 9. FACT CHECKER NODE
# ------------------------------------------------------------------
def fact_checker_node(state: State) -> dict:
    _emit(_job(state), "fact_checker", "started", "Auditing claims for accuracy...")
    logger.info("🕵️ FACT CHECKING ---")
    checker = llm_quality.with_structured_output(FactCheckReport)
    
    evidence_summary = "\n".join([
        f"- {e.title} ({e.url})\n  Content: {e.snippet[:500]}..."
        for e in state.get("evidence", [])[:15]
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
    
    logger.info(f"📊 Score: {report.score}/10 | Verdict: {report.verdict}")
    _emit(_job(state), "fact_checker", "completed", f"Score: {report.score}/10 — {report.verdict}", {"score": report.score, "verdict": report.verdict, "issues": len(report.issues) if report.issues else 0})
    
    # Store structured data for revision loop
    issues_list = [
        {
            "claim": issue.claim,
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "recommendation": issue.recommendation,
        }
        for issue in report.issues
    ] if report.issues else []
    
    return {
        "fact_check_report": report_text,
        "fact_check_verdict": report.verdict,
        "fact_check_issues": issues_list,
        "fact_check_score": report.score,
    }

# ------------------------------------------------------------------
# 9b. REVISION NODE (SELF-HEALING FACT-CHECK LOOP)
# ------------------------------------------------------------------
def revision_node(state: State) -> dict:
    """Fixes flagged claims from the fact-checker using the REVISION_SYSTEM prompt."""
    attempts = state.get("fact_check_attempts", 0)
    _emit(_job(state), "revision", "started", f"Self-healing revision (attempt {attempts + 1})...")
    logger.info(f"🔧 REVISING CONTENT (Attempt {attempts + 1})")
    
    issues = state.get("fact_check_issues", [])
    if not issues:
        logger.warning("No issues to fix, skipping revision.")
        return {}
    
    # Format issues for the LLM
    issues_text = "\n".join([
        f"{i+1}. [{iss['issue_type']}] \"{iss['claim']}\"\n   Fix: {iss['recommendation']}"
        for i, iss in enumerate(issues)
    ])
    
    # Build evidence context for the revision agent
    evidence = state.get("evidence", [])
    evidence_text = "\n".join([
        f"- [{e.title}]({e.url})\n  Content: {e.snippet[:400]}"
        for e in evidence[:10]
    ])
    
    response = llm_quality.invoke([
        SystemMessage(content=REVISION_SYSTEM),
        HumanMessage(content=(
            f"ORIGINAL BLOG:\n{state['final']}\n\n"
            f"FLAGGED ISSUES ({len(issues)} total):\n{issues_text}\n\n"
            f"AVAILABLE EVIDENCE (use for citations):\n{evidence_text}"
        )),
    ], max_tokens=8000)
    
    revised = response.content.strip()
    
    # Basic sanity check: don't accept a drastically shorter revision
    original_words = len(state.get("final", "").split())
    revised_words = len(revised.split())
    
    if revised_words < (original_words * 0.7):
        logger.warning(f"Revision too short ({revised_words} vs {original_words} words), keeping original.")
        return {"fact_check_attempts": attempts + 1}
    
    logger.info(f"✅ Revised: {revised_words} words (was {original_words})")
    _emit(_job(state), "revision", "completed", f"Revised: {revised_words} words", {"words": revised_words})
    return {
        "final": revised,
        "fact_check_attempts": attempts + 1,
    }

# ------------------------------------------------------------------
# 10. CAMPAIGN GENERATOR NODE
# ------------------------------------------------------------------
def campaign_generator_node(state: State) -> dict:
    _emit(_job(state), "campaign_generator", "started", "Generating 6-part omnichannel campaign...")
    logger.info("🚀 GENERATING OMNICHANNEL CAMPAIGN PACK ---")
    
    from Graph.templates import (
        LINKEDIN_SYSTEM, YOUTUBE_SYSTEM, FACEBOOK_SYSTEM,
        EMAIL_SEQUENCE_SYSTEM, TWITTER_THREAD_SYSTEM, LANDING_PAGE_SYSTEM
    )
    
    blog_post = state["final"]
    topic = state["topic"]
    evidence = state.get("evidence", [])
    
    key_stats = "\n".join([f"- {e.snippet[:100]}... ({e.url})" for e in evidence[:5]])
    
    # Construct Context Once
    context = f"TOPIC: {topic}\nBLOG CONTENT:\n{blog_post[:4000]}\nSTATS:\n{key_stats}"
    
    # Parallel generation — all 6 platforms at once
    def _gen(system_prompt):
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)]).content
    
    with ThreadPoolExecutor(max_workers=6) as pool:
        linkedin_future = pool.submit(_gen, LINKEDIN_SYSTEM)
        youtube_future  = pool.submit(_gen, YOUTUBE_SYSTEM)
        facebook_future = pool.submit(_gen, FACEBOOK_SYSTEM)
        email_future    = pool.submit(_gen, EMAIL_SEQUENCE_SYSTEM)
        twitter_future  = pool.submit(_gen, TWITTER_THREAD_SYSTEM)
        landing_future  = pool.submit(_gen, LANDING_PAGE_SYSTEM)
    
    linkedin = linkedin_future.result()
    youtube  = youtube_future.result()
    facebook = facebook_future.result()
    email    = email_future.result()
    twitter  = twitter_future.result()
    landing  = landing_future.result()
    
    logger.info("✅ All 6 Campaign Assets Generated")
    _emit(_job(state), "campaign_generator", "completed", "Generated Email, Twitter, Landing Page, LinkedIn, YouTube & Facebook content", {"assets": 6})
    
    return {
        "linkedin_post": linkedin,
        "youtube_script": youtube,
        "facebook_post": facebook,
        "email_sequence": email,
        "twitter_thread": twitter,
        "landing_page": landing,
    }

# ------------------------------------------------------------------
# 11. EVALUATOR NODE
# ------------------------------------------------------------------
def evaluator_node(state: State) -> dict:
    _emit(_job(state), "evaluator", "started", "Scoring final blog quality...")
    logger.info("📊 EVALUATING QUALITY ---")
    try:
        from validators import BlogEvaluator
        evaluator = BlogEvaluator()
        results = evaluator.evaluate(blog_post=state["final"], topic=state["topic"])
        score = results.get('final_score', 0)
        logger.info(f"🏆 Final Score: {score}/10")
        _emit(_job(state), "evaluator", "completed", f"Quality Score: {score}/10", {"score": score})
        return {"quality_evaluation": results}
    except ImportError:
        logger.warning("Validators module not found, skipping evaluation.")
        _emit(_job(state), "evaluator", "completed", "Evaluation skipped (module missing)")
        return {"quality_evaluation": {"error": "Module missing"}}
    except Exception as e:
        logger.warning(f"Evaluation Error: {e}")
        _emit(_job(state), "evaluator", "error", f"Evaluation failed: {str(e)}")
        return {"quality_evaluation": {"error": str(e)}}