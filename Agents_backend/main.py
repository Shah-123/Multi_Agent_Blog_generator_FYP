import os
import sys
from datetime import date
from dotenv import load_dotenv

# Environment Setup
load_dotenv()

# LangGraph Imports - NOW INCLUDES MEMORY & CHECKPOINTING
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
    evaluator_node
)
from validators import TopicValidator

# ===========================================================================
# HELPER: PLAN EDITOR (For Human Feedback)
# ===========================================================================
def refine_plan_with_llm(current_plan: Plan, feedback: str) -> Plan:
    """
    Uses AI to modify the plan based on user feedback.
    This runs 'outside' the graph, just to update the state.
    """
    print(f"\n   🤖 Refining plan based on: '{feedback}'...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # We define a simple structured output for the updated plan
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
# BUILD GRAPH (WITH INTERRUPTS)
# ===========================================================================
def build_graph(memory=None): # <--- THIS IS THE CRITICAL FIX
    """
    Constructs the workflow. 
    Accepts 'memory' to persist state across Streamlit re-runs.
    """

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
    
    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reducer", reducer.compile()) 
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("social_media", social_media_node)
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
    workflow.add_edge("social_media", "evaluator")
    workflow.add_edge("evaluator", END)

    # 3. Add Checkpointer (Memory)
    # If no memory provided (first run), create new one.
    # If memory provided (Streamlit re-run), use it.
    if memory is None:
        memory = MemorySaver()
    
    return workflow.compile(
        checkpointer=memory, 
        interrupt_after=["orchestrator"]
    )
# ===========================================================================
# MAIN RUNNER (HUMAN-IN-THE-LOOP)
# ===========================================================================
def run_app():
    print("="*80)
    print("🚀 AI CONTENT FACTORY (HUMAN-IN-THE-LOOP MODE)")
    print("="*80)
    
    # 1. Get Topic
    topic = input("\n📝 Enter blog topic: ").strip()
    if not topic: return
    
    validator = TopicValidator()
    if not validator.validate(topic)["valid"]:
        print("❌ Topic Rejected.")
        return

    # 2. Config for Persistence (Required for HITL)
    # We use a static thread_id so we can resume the same session
    thread_config = {"configurable": {"thread_id": "1"}}
    
    initial_state = {
        "topic": topic,
        "as_of": date.today().isoformat(),
        "sections": [],
    }
    
    print(f"\n🚀 PHASE 1: RESEARCH & PLANNING...")
    print("-" * 60)
    
    app = build_graph()
    
    # -----------------------------------------------------------------------
    # PASS 1: Run until the Interrupt (After Orchestrator)
    # -----------------------------------------------------------------------
    # We use .stream() to see progress
    for event in app.stream(initial_state, thread_config, stream_mode="values"):
        # Just loop to let it run. The print statements in nodes.py will show progress.
        pass

    # -----------------------------------------------------------------------
    # INTERMISSION: HUMAN APPROVAL
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("✋ PAUSED FOR HUMAN REVIEW")
    print("="*60)
    
    # Fetch the state (which is currently paused at Orchestrator)
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
            
            # CALL HELPER TO UPDATE PLAN
            new_plan = refine_plan_with_llm(plan, feedback)
            
            # UPDATE GRAPH STATE
            app.update_state(thread_config, {"plan": new_plan})
            
            print(f"\n✅ Plan Updated: '{new_plan.blog_title}'")
            # Show new sections briefly
            for t in new_plan.tasks:
                 print(f"   - {t.title}")
            
            confirm = input("\nProceed with this new plan? (y/n): ").lower()
            if confirm == 'y':
                break
            # If 'n', loop continues
        else:
            print("Please enter 'y' or 'n'.")

    # -----------------------------------------------------------------------
    # PASS 2: Resume Execution (Fanout -> Writers -> End)
    # -----------------------------------------------------------------------
    print(f"\n🚀 PHASE 2: WRITING & POLISHING...")
    print("-" * 60)
    
    # Passing None resumes execution from where it paused
    # Recursion limit high for map-reduce
    for event in app.stream(None, thread_config, stream_mode="values", recursion_limit=100):
        pass

    # -----------------------------------------------------------------------
    # FINAL DISPLAY
    # -----------------------------------------------------------------------
    final_state = app.get_state(thread_config).values
    
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE")
    print("="*80)
    
    if final_state.get("final"):
        print(f"\n📄 BLOG PREVIEW:\n{final_state['final'][:500]}...\n")
        print(f"👉 Full content saved to file.")
    
    if final_state.get("linkedin_post"):
        print(f"\n📱 LINKEDIN PREVIEW:\n{final_state['linkedin_post'][:200]}...")

    if final_state.get("fact_check_report"):
         print(f"\n🕵️ FACT CHECK:\n{final_state['fact_check_report'][:200]}...")

if __name__ == "__main__":
    run_app()