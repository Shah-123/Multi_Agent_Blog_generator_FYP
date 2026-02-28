import os
import sys
import json
import re
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# Environment Setup
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found in .env file.")
    sys.exit(1)

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Internal Imports
from Graph.state import State, Plan
from Graph.nodes import (
    router_node, 
    research_node, 
    orchestrator_node, 
    worker_node, 
    fanout, 
    merge_content, 
    decide_images, 
    generate_and_place_images,
    qa_agent_node,
    campaign_generator_node, 
    _safe_slug
)
from Graph.keyword_optimizer import keyword_optimizer_node
from validators import TopicValidator

# ===========================================================================
# PODCAST IMPORT WITH FALLBACK
# ===========================================================================
try:
    from Graph.podcast_studio import podcast_node
    PODCAST_AVAILABLE = True
except ImportError:
    PODCAST_AVAILABLE = False
    def podcast_node(state: dict) -> dict:
        return {"audio_path": None, "script_path": None}

# ===========================================================================
# 1. HELPER FUNCTIONS
# ===========================================================================

def create_blog_structure(topic: str) -> dict:
    """Creates organized folder structure for the blog."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = _safe_slug(topic)[:50]
    
    base_folder = f"blogs/{safe_topic}_{timestamp}"
    
    folders = {
        "base": base_folder,
        "content": f"{base_folder}/content",
        "social": f"{base_folder}/social_media",
        "reports": f"{base_folder}/reports",
        "assets": f"{base_folder}/assets/images",
        "research": f"{base_folder}/research",
        "audio": f"{base_folder}/audio",
        "metadata": f"{base_folder}/metadata"
    }
    
    for path in folders.values():
        Path(path).mkdir(parents=True, exist_ok=True)
        
    return folders

def refine_plan_with_llm(current_plan: Plan, feedback: str) -> Plan:
    """Refines the plan based on human feedback."""
    print(f"\n   🤖 Refining plan based on: '{feedback}'...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    editor = llm.with_structured_output(Plan)
    
    return editor.invoke([
        SystemMessage(content="You are a helpful editor. Update the Plan based STRICTLY on user feedback."),
        HumanMessage(content=f"Current Plan:\n{current_plan.model_dump_json()}\n\nFeedback: {feedback}")
    ])

def generate_readme(folders: dict, saved_files: dict, state: State) -> str:
    """Generates a README.md file summarizing the project generation."""
    topic = state.get("topic", "Unknown Topic")
    tone = state.get("target_tone", "N/A")
    keywords = state.get("target_keywords", [])
    score = state.get("qa_score", "N/A")
    word_count = len(state.get("final", "").split())
    
    # Get blog filename
    blog_file = os.path.basename(saved_files.get("blog", "blog.md"))
    
    # Safely access Plan object
    plan = state.get("plan")
    if plan:
        audience = plan.audience if hasattr(plan, 'audience') else "General"
    else:
        audience = "General"
    
    md = f"""# {topic}

## 📋 Blog Information
- **Topic**: {topic}
- **Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Target Audience**: {audience}
- **Tone**: {tone}
- **Word Count**: {word_count}

## 📁 Folder Structure
```
{os.path.basename(folders['base'])}/
├── content/
│   └── {blog_file}                    # Main blog post with images
├── social_media/
│   ├── linkedin_*.txt
│   ├── youtube_*.txt
│   └── facebook_*.txt
├── reports/
│   ├── qa_report.txt
│   └── keyword_optimization.txt
├── research/
│   └── evidence.json
├── assets/
│   └── images/                        # Generated images
├── audio/                             # Podcast (if generated)
└── metadata/
    ├── plan.json
    └── metadata.json
```

## 📊 Quality Metrics

{state.get('qa_report', 'No QA report available')}

---

### Quality Score: {score}/10

## 🎯 SEO Details
- **Target Keywords**: {', '.join(keywords) if keywords else 'None specified'}
- **Tone**: {tone}
- **Mode**: {state.get("mode")}
- **Evidence Sources**: {len(state.get("evidence", []))}

## 🚀 How to Use
1. **Main Blog**: Open `content/{blog_file}` in any markdown viewer
2. **Social Media**: Copy content from `social_media/` files
3. **Reports**: Check `reports/` for quality analysis

## ⚙️ Generation Details
- **Mode**: {state.get("mode")}
- **Research Queries**: {len(state.get("queries", []))}
- **Evidence Sources**: {len(state.get("evidence", []))}

---
*Generated by AI Content Factory*
"""
    readme_path = f"{folders['base']}/README.md"
    Path(readme_path).write_text(md, encoding="utf-8")
    return readme_path

def save_blog_content(folders: dict, state: State) -> dict:
    """Saves all outputs to their respective folders."""
    saved = {}
    plan = state.get("plan")
    if not plan: return saved

    slug = _safe_slug(plan.blog_title)

    # 1. Content - ONLY SAVE THE FINAL VERSION WITH IMAGES
    if state.get("final"):
        path = f"{folders['content']}/{slug}.md"
        Path(path).write_text(state["final"], encoding="utf-8")
        saved["blog"] = path
        print(f"   ✅ Saved blog: {os.path.basename(path)}")

    # 2. Campaign Assets
    for platform in ["linkedin", "facebook", "youtube", "twitter", "email", "landing_page"]:
        # Match the state keys correctly
        if platform == "youtube":
            key = "youtube_script"
            ext = "txt"
        elif platform == "email":
            key = "email_sequence"
            ext = "md"
        elif platform == "twitter":
            key = "twitter_thread"
            ext = "md"
        elif platform == "landing_page":
            key = "landing_page"
            ext = "md"
        else:
            key = f"{platform}_post"
            ext = "txt"
            
        if state.get(key):
            path = f"{folders['social']}/{platform}_{slug}.{ext}"
            Path(path).write_text(state[key], encoding="utf-8")
            saved[platform] = path

    # 3. Reports
    if state.get("qa_report"):
        path = f"{folders['reports']}/qa_report.txt"
        Path(path).write_text(state["qa_report"], encoding="utf-8")
        saved["qa_report"] = path
    
    # Save keyword report
    if state.get("keyword_report"):
        path = f"{folders['reports']}/keyword_optimization.txt"
        Path(path).write_text(state["keyword_report"], encoding="utf-8")
        saved["keyword_report"] = path

    # 4. Audio
    if state.get("audio_path") and os.path.exists(state["audio_path"]):
        import shutil
        dest = f"{folders['audio']}/podcast.mp3"
        shutil.copy(state["audio_path"], dest)
        saved["audio"] = dest
        if state.get("script_path"):
            shutil.copy(state["script_path"], f"{folders['audio']}/script.txt")

    # 5. Research Evidence
    if state.get("evidence"):
        evidence_data = [e.model_dump() if hasattr(e, 'model_dump') else e 
                        for e in state["evidence"]]
        path = f"{folders['research']}/evidence.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(evidence_data, f, indent=2)
        saved["evidence"] = path

    # 6. Metadata
    meta = {
        "topic": state.get("topic"),
        "as_of": state.get("as_of"),
        "mode": state.get("mode"),
        "generated_at": datetime.now().isoformat(),
        "word_count": len(state.get("final", "").split()),
        "target_tone": state.get("target_tone"),
        "target_keywords": state.get("target_keywords", []),
        "file_paths": {
            "blog": saved.get("blog"),
            "assets": [saved.get(k) for k in ["linkedin", "youtube", "facebook", "twitter", "email", "landing_page"] if saved.get(k)],
            "qa_report": saved.get("qa_report"),
            "keyword_report": saved.get("keyword_report"),
            "evidence": saved.get("evidence"),
            "plan": f"{folders['metadata']}/plan.json"
        }
    }
    
    with open(f"{folders['metadata']}/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    
    if plan:
        with open(f"{folders['metadata']}/plan.json", 'w', encoding='utf-8') as f:
            json.dump(plan.model_dump(), f, indent=2)

    return saved

# ===========================================================================
# 2. BUILD GRAPH
# ===========================================================================
def build_graph(memory=None):
    """Build the LangGraph workflow with all nodes."""
    
    if memory is None:
        memory = MemorySaver()
    
    # Subgraph for Reducer
    reducer = StateGraph(State)
    reducer.add_node("merge_content", merge_content)
    reducer.add_node("decide_images", decide_images)
    reducer.add_node("generate_and_place_images", generate_and_place_images)
    reducer.add_edge(START, "merge_content")
    
    # Conditional edge to skip image generation if disabled
    reducer.add_conditional_edges("merge_content",
        lambda s: "decide_images" if s.get("generate_images", True) else END
    )
    
    reducer.add_edge("decide_images", "generate_and_place_images")
    reducer.add_edge("generate_and_place_images", END)

    # Main Graph
    workflow = StateGraph(State)
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reducer", reducer.compile())
    workflow.add_node("qa_agent", qa_agent_node)
    workflow.add_node("keyword_optimizer", keyword_optimizer_node)
    workflow.add_node("campaign_generator", campaign_generator_node)
    
    if PODCAST_AVAILABLE:
        workflow.add_node("audio_generator", podcast_node)
    else:
        workflow.add_node("audio_generator", lambda s: {})

    # Edges
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges("router",
        lambda s: "research" if s["needs_research"] else "orchestrator"
    )
    workflow.add_edge("research", "orchestrator")
    workflow.add_conditional_edges("orchestrator", fanout, ["worker"])

    # FIX 1: workers → reducer first (merge all sections + place images)
    # Previously this was: workflow.add_edge("worker", "fact_checker")  ← BUG
    workflow.add_edge("worker", "reducer")

    # FIX 2: reducer → qa_agent
    workflow.add_edge("reducer", "qa_agent")

    # FIX 3: qa_agent → keyword_optimizer (No self-healing loop)
    workflow.add_edge("qa_agent", "keyword_optimizer")
    
    # Cost-saving conditional routing
    def after_keyword_router(s):
        if s.get("generate_campaign", True):
            return "campaign_generator"
        elif s.get("generate_audio", True) and PODCAST_AVAILABLE:
            return "audio_generator"
        return END
        
    workflow.add_conditional_edges("keyword_optimizer", after_keyword_router, 
        ["campaign_generator", "audio_generator", END]
    )
    
    def after_campaign_router(s):
        if s.get("generate_audio", True) and PODCAST_AVAILABLE:
            return "audio_generator"
        return END
        
    workflow.add_conditional_edges("campaign_generator", after_campaign_router,
        ["audio_generator", END]
    )
    
    workflow.add_edge("audio_generator", END)

    return workflow.compile(
        checkpointer=memory,
        interrupt_after=["orchestrator"]
    )

# ===========================================================================
# 3. MAIN RUNNER
# ===========================================================================
def run_app():
    print("="*80)
    print("🚀 AI CONTENT FACTORY (FYP EDITION)")
    print("="*80)
    
    # 1. Input & Validation
    topic = input("\n📝 Enter blog topic: ").strip()
    if not topic: return
    
    valid = TopicValidator().validate(topic)
    if not valid["valid"]:
        print(f"❌ Rejected: {valid['reason']}")
        return
    
    print(f"✅ Topic Accepted: {topic}")
    
    # 2. Get Tone
    print("\n🎨 Select Tone:")
    print("1. Professional (formal, data-driven)")
    print("2. Conversational (friendly, relatable)")
    print("3. Technical (precise, expert-level)")
    print("4. Educational (teaching-focused)")
    print("5. Persuasive (compelling, action-driven)")
    print("6. Inspirational (motivating, aspirational)")
    
    tone_map = {
        "1": "professional",
        "2": "conversational",
        "3": "technical",
        "4": "educational",
        "5": "persuasive",
        "6": "inspirational"
    }
    
    tone_choice = input("Choose (1-6) [default: 1]: ").strip() or "1"
    target_tone = tone_map.get(tone_choice, "professional")
    
    # 3. Get Keywords
    keywords_input = input("\n🎯 Enter target keywords (comma-separated, or press Enter to skip): ").strip()
    target_keywords = [k.strip() for k in keywords_input.split(",")] if keywords_input else []
    
    print("\n💰 Cost-Saving Options (Press Enter for Yes):")
    gen_img_input = input("Generate Images (Gemini)? [Y/n]: ").strip().lower()
    generate_images = gen_img_input != "n"
    
    gen_camp_input = input("Generate Social Media Campaign? [Y/n]: ").strip().lower()
    generate_campaign = gen_camp_input != "n"
    
    gen_aud_input = input("Generate Podcast Audio (TTS)? [Y/n]: ").strip().lower()
    generate_audio = gen_aud_input != "n"
    
    # 4. Get Number of Sections
    sections_input = input("\n📏 How many body sections should the blog have? (1-10) [default: 5]: ").strip()
    try:
        target_sections = int(sections_input) if sections_input else 5
        target_sections = max(1, min(10, target_sections)) # Clamp between 1 and 10
    except ValueError:
        target_sections = 5
        
    print(f"\n✅ Tone: {target_tone}")
    print(f"✅ Sections: {target_sections}")
    print(f"✅ Options: Images={'ON' if generate_images else 'OFF'} | Campaign={'ON' if generate_campaign else 'OFF'} | Audio={'ON' if generate_audio else 'OFF'}")
    print(f"✅ Keywords: {', '.join(target_keywords) if target_keywords else 'None specified'}")
    
    # 5. Setup Folders
    folders = create_blog_structure(topic)
    print(f"📁 Working Directory: {folders['base']}")
    
    # 5. Graph Config
    app = build_graph()
    thread = {"configurable": {"thread_id": f"job_{datetime.now().strftime('%M%S')}"}}
    
    initial_state = {
        "topic": topic,
        "as_of": date.today().isoformat(),
        "sections": [],
        "blog_folder": folders["base"],
        "target_tone": target_tone,
        "target_keywords": target_keywords,
        "target_sections": target_sections,
        "generate_images": generate_images,
        "generate_campaign": generate_campaign,
        "generate_audio": generate_audio
    }
    
    # 6. Phase 1: Research & Planning
    print("\n🚀 PHASE 1: RESEARCH & PLANNING")
    for _ in app.stream(initial_state, thread, stream_mode="values"): pass
    
    # 7. Human-in-the-Loop Review
    state = app.get_state(thread).values
    plan = state.get("plan")
    
    print("\n" + "="*60)
    print(f"📋 DRAFT PLAN: {plan.blog_title}")
    print(f"🎨 Tone: {plan.tone}")
    if plan.primary_keywords:
        print(f"🎯 Keywords: {', '.join(plan.primary_keywords)}")
    print("="*60)
    
    for t in plan.tasks:
        keyword_tags = f" [{', '.join(t.tags[:2])}]" if t.tags else ""
        print(f"   {t.id+1}. {t.title}{keyword_tags}")
    
    while True:
        feedback = input("\n✅ Approved? (y/n): ").lower()
        if feedback == 'y': 
            break
        elif feedback == 'n':
            notes = input("💬 Enter changes: ")
            new_plan = refine_plan_with_llm(plan, notes)
            app.update_state(thread, {"plan": new_plan})
            plan = new_plan
            print("\n✅ Plan Updated:")
            for t in plan.tasks: 
                print(f"   - {t.title}")
    
    # 8. Phase 2: Execution
    print("\n🚀 PHASE 2: WRITING & PRODUCTION")
    for _ in app.stream(None, thread, stream_mode="values", recursion_limit=150): pass
    
    # 9. Final Saving
    final_state = app.get_state(thread).values
    print("\n💾 SAVING ASSETS...")
    saved_files = save_blog_content(folders, final_state)
    readme = generate_readme(folders, saved_files, final_state)
    
    print("\n" + "="*80)
    print("✨ GENERATION COMPLETE ✨")
    print(f"📂 Output Folder: {folders['base']}")
    print(f"📖 Read Summary: {readme}")
    
    # Display keyword report if available
    if final_state.get("keyword_report"):
        print("\n" + "="*60)
        print(final_state["keyword_report"])
    
    print("="*80)

if __name__ == "__main__":
    run_app()