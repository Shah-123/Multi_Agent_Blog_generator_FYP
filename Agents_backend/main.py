import os
import sys
from datetime import date
from dotenv import load_dotenv

# Environment Setup
load_dotenv()

from langgraph.graph import StateGraph, START, END

# Import NEW State and Nodes (Candidate C Architecture)
from Graph.state import State
from Graph.nodes import (
    router_node, 
    research_node, 
    orchestrator_node, 
    worker_node, 
    fanout, 
    merge_content, 
    decide_images, 
    generate_and_place_images
)

# Keep your Validator (It's good)
from validators import TopicValidator

# ---------------------------------------------------------------------------
# GRAPH BUILDER (The Teacher's Architecture)
# ---------------------------------------------------------------------------
def build_graph():
    """Builds the Parallel Map-Reduce Graph."""
    
    # 1. Define the Reducer Subgraph (Merge -> Image Plan -> Generate)
    reducer = StateGraph(State)
    reducer.add_node("merge_content", merge_content)
    reducer.add_node("decide_images", decide_images)
    reducer.add_node("generate_and_place_images", generate_and_place_images)
    
    reducer.add_edge(START, "merge_content")
    reducer.add_edge("merge_content", "decide_images")
    reducer.add_edge("decide_images", "generate_and_place_images")
    reducer.add_edge("generate_and_place_images", END)

    # 2. Define Main Graph
    workflow = StateGraph(State)
    
    workflow.add_node("router", router_node)
    workflow.add_node("research", research_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reducer", reducer.compile()) 

    # 3. Define Edges & Routing
    workflow.add_edge(START, "router")
    
    # Logic: If research needed -> Research Node, Else -> Orchestrator
    def route_next(state):
        return "research" if state["needs_research"] else "orchestrator"
        
    workflow.add_conditional_edges("router", route_next)
    workflow.add_edge("research", "orchestrator")
    
    # Logic: Fanout to multiple workers in PARALLEL
    workflow.add_conditional_edges("orchestrator", fanout, ["worker"])
    
    # Logic: All workers return to Reducer
    workflow.add_edge("worker", "reducer")
    workflow.add_edge("reducer", END)

    return workflow.compile()

# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------
def run_app():
    """Interactive CLI runner."""
    print("="*80)
    print("🚀 AI CONTENT FACTORY (PARALLEL ARCHITECTURE)")
    print("="*80)
    
    # 1. Get Input
    topic = input("\n📝 Enter blog topic: ").strip()
    
    # 2. Validate Topic
    validator = TopicValidator()
    validation = validator.validate(topic)
    if not validation["valid"]:
        print(f"\n❌ Topic Rejected: {validation['reason']}")
        return
    
    print(f"\n✅ Topic Accepted: {validation['reason']}")
    
    # 3. Initialize State (MUST MATCH Graph/state.py)
    # The Teacher's architecture requires 'as_of' and empty 'sections'
    initial_state = {
        "topic": topic,
        "as_of": date.today().isoformat(),
        "sections": [], # Important for operator.add to work
        
        # Optional: You can pass hint instructions in the topic string if needed
        # "mode": "hybrid" # You can force a mode here if you want
    }
    
    # 4. Run Workflow
    print(f"\n🚀 STARTING WORKFLOW...")
    print(f"   Topic: {topic}")
    print(f"   Date: {initial_state['as_of']}")
    print("="*80 + "\n")
    
    try:
        app = build_graph()
        # Recursion limit needs to be high for complex graphs, though Map-Reduce is efficient
        final_output = app.invoke(initial_state, {"recursion_limit": 50})
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        return
    
    # 5. Display Results
    print("\n" + "="*80)
    print("✅ WORKFLOW COMPLETE")
    print("="*80)
    
    # The final Markdown is stored in the 'final' key (from generate_and_place_images)
    if final_output.get("final"):
        print("\n📝 FINAL BLOG POST PREVIEW:")
        print("-" * 40)
        # Print first 2000 chars to avoid flooding terminal
        print(final_output["final"][:2000] + "\n\n... (Check generated file for full content) ...")
        
        print("-" * 40)
        print(f"👉 Full file saved as .md in directory.")
        
        # Show Image Plan if available
        if final_output.get("image_specs"):
            print(f"\n🎨 Generated {len(final_output['image_specs'])} Images:")
            for img in final_output["image_specs"]:
                print(f"   - {img['filename']}: {img['prompt'][:50]}...")
    
    else:
        print("⚠️ No output generated.")

# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_app()