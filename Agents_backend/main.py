import os
import sys
from datetime import date, datetime
from pathlib import Path
from dotenv import load_dotenv

# Environment Setup
load_dotenv()

# LangGraph Imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Import State & Models
from Graph.state import State, Plan
from Graph.structured_data import FactCheckReport

# Import Nodes
from Graph.nodes import (
    router_node, 
    research_node, 
    orchestrator_node, 
    worker_node, 
    fanout, 
    merge_content, 
    decide_images, 
    generate_and_place_images,
    fact_checker_node, 
    social_media_node, 
    evaluator_node,
    _safe_slug
)
from validators import TopicValidator

# ===========================================================================
# HELPER: PLAN EDITOR (For Human Feedback)
# ===========================================================================
def refine_plan_with_llm(current_plan: Plan, feedback: str) -> Plan:
    """
    Uses AI to modify the plan based on user feedback.
    """
    print(f"\n   🤖 Refining plan based on: '{feedback}'...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    editor = llm.with_structured_output(Plan)
    
    system_prompt = """You are a helpful editor. 
    Update the provided Blog Plan based STRICTLY on the user's feedback.
    - If they say "Add section X", add it.
    - If they say "Remove section Y", remove it.
    - Keep the rest of the plan consistent.
    """
    
    new_plan = editor.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Current Plan JSON:\n{current_plan.model_dump_json()}\n\nUser Feedback: {feedback}")
    ])
    
    return new_plan

# ===========================================================================
# FILE SYSTEM ORGANIZATION FUNCTIONS
# ===========================================================================
def create_blog_structure(topic: str) -> dict:
    """
    Creates organized folder structure for a blog.
    Returns: dict with paths for different content types
    """
    # Create timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = _safe_slug(topic)[:50]  # Limit length
    
    # Base folder
    base_folder = f"blogs/{safe_topic}_{timestamp}"
    
    # Subfolders
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
    
    # Create all folders
    for folder_path in folders.values():
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        print(f"   📁 Created: {folder_path}")
    
    return folders

def save_blog_content(folders: dict, state: State) -> dict:
    """
    Save all blog components to organized folders.
    Returns: dict of saved file paths
    """
    saved_files = {}
    plan = state.get("plan")
    
    if not plan:
        print("❌ No plan found to save content.")
        return saved_files
    
    # 1. Save main blog markdown
    if state.get("final"):
        blog_filename = f"{folders['content']}/{_safe_slug(plan.blog_title)}.md"
        Path(blog_filename).write_text(state["final"], encoding="utf-8")
        saved_files["blog"] = blog_filename
        print(f"   📄 Blog saved: {blog_filename}")
    
    # 2. Save blog without images (clean version)
    if state.get("merged_md"):
        clean_filename = f"{folders['content']}/{_safe_slug(plan.blog_title)}_no_images.md"
        Path(clean_filename).write_text(state["merged_md"], encoding="utf-8")
        saved_files["blog_clean"] = clean_filename
    
    # 3. Save social media posts
    social_files = []
    for platform, content in [("linkedin", state.get("linkedin_post")),
                             ("youtube", state.get("youtube_script")),
                             ("facebook", state.get("facebook_post"))]:
        if content:
            filename = f"{folders['social']}/{platform}_{_safe_slug(plan.blog_title)}.txt"
            Path(filename).write_text(content, encoding="utf-8")
            social_files.append(filename)
            print(f"   📱 {platform.capitalize()} saved: {filename}")
    
    saved_files["social"] = social_files
    
    # 4. Save reports
    if state.get("fact_check_report"):
        report_file = f"{folders['reports']}/fact_check.txt"
        Path(report_file).write_text(state["fact_check_report"], encoding="utf-8")
        saved_files["fact_check"] = report_file
        print(f"   🕵️ Fact check saved: {report_file}")
    
    if state.get("quality_evaluation"):
        import json
        eval_file = f"{folders['reports']}/quality_evaluation.json"
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(state["quality_evaluation"], f, indent=2)
        saved_files["quality_eval"] = eval_file
        print(f"   📊 Quality eval saved: {eval_file}")
    
    # 5. Save research evidence
    if state.get("evidence"):
        import json
        evidence_file = f"{folders['research']}/evidence.json"
        evidence_data = [e.model_dump() for e in state["evidence"]]
        with open(evidence_file, 'w', encoding='utf-8') as f:
            json.dump(evidence_data, f, indent=2)
        saved_files["evidence"] = evidence_file
        print(f"   🔍 Research evidence saved: {evidence_file}")
    
    # 6. Save plan
    if plan:
        plan_file = f"{folders['metadata']}/plan.json"
        with open(plan_file, 'w', encoding='utf-8') as f:
            f.write(plan.model_dump_json(indent=2))
        saved_files["plan"] = plan_file
    
    # 7. Save metadata
    metadata = {
        "topic": state.get("topic"),
        "as_of": state.get("as_of"),
        "mode": state.get("mode"),
        "generated_at": datetime.now().isoformat(),
        "word_count": len(state.get("final", "").split()) if state.get("final") else 0,
        "file_paths": saved_files
    }
    
    metadata_file = f"{folders['metadata']}/metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    saved_files["metadata"] = metadata_file
    
    return saved_files

def generate_readme(folders: dict, saved_files: dict, state: State) -> str:
    """Generate a README file for the blog package."""
    plan = state.get("plan")
    
    readme_content = f"""# {plan.blog_title if plan else 'Generated Blog'}

## 📋 Blog Information
- **Topic**: {state.get('topic', 'N/A')}
- **Generated Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Target Audience**: {plan.audience if plan else 'N/A'}
- **Tone**: {plan.tone if plan else 'N/A'}
- **Word Count**: {len(state.get('final', '').split()) if state.get('final') else 0}

## 📁 Folder Structure

## 📊 Quality Metrics
{state.get('fact_check_report', 'No fact check available')}

## 🚀 How to Use
1. **Main Blog**: Open `content/{_safe_slug(plan.blog_title)}.md` in any markdown viewer
2. **Social Media**: Copy content from `social_media/` files
3. **Reports**: Check `reports/` for quality analysis

## ⚙️ Generation Details
- **Mode**: {state.get('mode', 'N/A')}
- **Research Queries**: {len(state.get('queries', []))}
- **Evidence Sources**: {len(state.get('evidence', []))}

---
*Generated by AI Content Factory*
"""
    
    readme_file = f"{folders['base']}/README.md"
    Path(readme_file).write_text(readme_content, encoding="utf-8")
    return readme_file

# ===========================================================================
# BUILD GRAPH (WITH INTERRUPTS)
# ===========================================================================
def build_graph(memory=None):
    """Constructs the workflow."""
    # 1. Reducer Subgraph
    reducer = StateGraph(State)
    reducer.add_node("merge_content", merge_content)
    reducer.add_node("decide_images", decide_images)
    reducer.add_node("generate_and_place_images", generate_and_place_images)
    reducer.add_edge(START, "merge_content")
    reducer.add_edge("merge_content", "decide_images")
    reducer.add_edge("decide_images", "generate_and_place_images")
    reducer.add_edge("generate_and_place_images", END)

    # 2. Main Graph
    workflow = StateGraph(State)
    from Graph.podcast_studio import podcast_node

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reducer", reducer.compile()) 
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("social_media", social_media_node)
    workflow.add_node("audio_generator", podcast_node)
    workflow.add_node("evaluator", evaluator_node)

    # Edges
    workflow.add_edge(START, "router")
    
    def route_next(state):
        return "research" if state["needs_research"] else "orchestrator"
        
    workflow.add_conditional_edges("router", route_next)
    workflow.add_edge("research", "orchestrator")
    workflow.add_conditional_edges("orchestrator", fanout, ["worker"])
    workflow.add_edge("worker", "reducer")
    workflow.add_edge("reducer", "fact_checker")
    workflow.add_edge("fact_checker", "social_media")
    workflow.add_edge("social_media", "audio_generator")
    workflow.add_edge("audio_generator", "evaluator")
    workflow.add_edge("evaluator", END)

    # 3. Add Checkpointer
    if memory is None:
        memory = MemorySaver()
    
    return workflow.compile(
        checkpointer=memory, 
        interrupt_after=["orchestrator"]
    )

# ===========================================================================
# MAIN RUNNER WITH ORGANIZED STORAGE
# ===========================================================================
def run_app():
    print("="*80)
    print("🚀 AI CONTENT FACTORY (ORGANIZED STORAGE MODE)")
    print("="*80)
    
    # 1. Get Topic
    topic = input("\n📝 Enter blog topic: ").strip()
    if not topic: 
        print("❌ No topic provided.")
        return
    
    # 2. Validate topic
    validator = TopicValidator()
    validation_result = validator.validate(topic)
    
    if not validation_result["valid"]:
        print(f"❌ Topic Rejected: {validation_result['reason']}")
        return
    
    print(f"✅ Topic validated: {topic}")
    
    # 3. Create organized folder structure
    print(f"\n📁 Creating organized folder structure...")
    folders = create_blog_structure(topic)
    print(f"   ✅ Base folder: {folders['base']}")
    
    # 4. Initial state with folder info
    thread_config = {"configurable": {"thread_id": "1"}}
    
    initial_state = {
        "topic": topic,
        "as_of": date.today().isoformat(),
        "sections": [],
        "blog_folder": folders["base"]  # Pass folder info to nodes
    }
    
    print(f"\n🚀 PHASE 1: RESEARCH & PLANNING...")
    print("-" * 60)
    
    app = build_graph()
    
    # -----------------------------------------------------------------------
    # PASS 1: Run until the Interrupt
    # -----------------------------------------------------------------------
    for event in app.stream(initial_state, thread_config, stream_mode="values"):
        pass

    # -----------------------------------------------------------------------
    # HUMAN REVIEW
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("✋ PAUSED FOR HUMAN REVIEW")
    print("="*60)
    
    current_state = app.get_state(thread_config).values
    plan: Plan = current_state.get("plan")
    
    if not plan:
        print("❌ Error: No plan found. Exiting.")
        return

    # Display the Plan
    print(f"\n📋 PROPOSED PLAN FOR: '{plan.blog_title}'")
    print(f"   Target Audience: {plan.audience} | Tone: {plan.tone}")
    print("-" * 40)
    for t in plan.tasks:
        print(f"   {t.id}. {t.title}")
    print("-" * 40)
    
    # Ask for feedback
    while True:
        choice = input("\nDoes this look good? (y/n): ").lower().strip()
        
        if choice == 'y':
            print("\n✅ Plan Approved. Resuming generation...")
            break
        elif choice == 'n':
            feedback = input("📝 Enter your feedback (e.g., 'Add a section on Ethics'): ")
            
            new_plan = refine_plan_with_llm(plan, feedback)
            app.update_state(thread_config, {"plan": new_plan})
            
            print(f"\n✅ Plan Updated: '{new_plan.blog_title}'")
            for t in new_plan.tasks:
                 print(f"   - {t.title}")
            
            confirm = input("\nProceed with this new plan? (y/n): ").lower()
            if confirm == 'y':
                break
        else:
            print("Please enter 'y' or 'n'.")

    # -----------------------------------------------------------------------
    # PASS 2: Resume Execution
    # -----------------------------------------------------------------------
    print(f"\n🚀 PHASE 2: WRITING & POLISHING...")
    print("-" * 60)
    
    for event in app.stream(None, thread_config, stream_mode="values", recursion_limit=100):
        pass

    # -----------------------------------------------------------------------
    # SAVE ALL CONTENT TO ORGANIZED FOLDERS
    # -----------------------------------------------------------------------
    final_state = app.get_state(thread_config).values
    
    print("\n" + "="*80)
    print("💾 SAVING ALL CONTENT TO ORGANIZED FOLDERS")
    print("="*80)
    
    # Save everything
    saved_files = save_blog_content(folders, final_state)
    
    # Generate README
    readme_file = generate_readme(folders, saved_files, final_state)
    
    # -----------------------------------------------------------------------
    # FINAL SUMMARY
    # -----------------------------------------------------------------------
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE - ALL CONTENTS ORGANIZED")
    print("="*80)
    
    if final_state.get("final"):
        full_blog = final_state['final']
        word_count = len(full_blog.split())
        
        print(f"\n📊 BLOG STATISTICS:")
        print(f"   📝 Title: {plan.blog_title}")
        print(f"   🎯 Audience: {plan.audience}")
        print(f"   🎨 Tone: {plan.tone}")
        print(f"   📈 Word Count: {word_count:,}")
        print(f"   ⏱️ Estimated Read Time: {word_count // 200} minutes")
        print(f"   📁 Saved to: {folders['base']}")
        
        # Show folder structure
        print(f"\n📁 FOLDER STRUCTURE CREATED:")
        for root, dirs, files in os.walk(folders['base']):
            level = root.replace(folders['base'], '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f'{subindent}{file}')
            if len(files) > 5:
                print(f'{subindent}... and {len(files)-5} more files')
        
        # Quick preview
        print(f"\n📄 BLOG PREVIEW (First 300 chars):")
        print("-" * 60)
        print(full_blog[:300] + "...")
        print("-" * 60)
        
        # Show saved files
        print(f"\n💾 FILES SAVED:")
        for file_type, file_path in saved_files.items():
            if isinstance(file_path, list):
                for f in file_path:
                    print(f"   📄 {os.path.basename(f)}")
            else:
                print(f"   📄 {os.path.basename(file_path)}")
        
        print(f"\n📖 Complete README: {readme_file}")
        print(f"\n🎯 NEXT STEPS:")
        print(f"   1. Open {saved_files.get('blog', 'blog.md')} to view the blog")
        print(f"   2. Check {folders['social']}/ for social media posts")
        print(f"   3. Review {saved_files.get('fact_check', 'fact_check.txt')} for quality report")
        
    else:
        print("❌ No blog content generated.")

    print("\n" + "="*80)
    print("✨ All content saved in organized folders! ✨")
    print("="*80)

if __name__ == "__main__":
    run_app()