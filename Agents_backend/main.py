import os
from dotenv import load_dotenv

# 1. Environment Setup
load_dotenv()

from langgraph.graph import StateGraph, END
from Graph.state import AgentState
from Graph.nodes import (
    researcher_node, 
    analyst_node, 
    writer_node, 
    fact_checker_node,
    # image_generator_node # Ensure this is imported
)
from validators import TopicValidator, realistic_evaluation

# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------

def evaluator_node(state: AgentState):
    """Evaluates the blog quality using our new NLI/Hallucination-aware logic."""
    print("--- 📊 EVALUATING QUALITY (NLI & Structure) ---")
    
    # Run the new evaluation logic from validators.py
    results = realistic_evaluation(
        blog_post=state["final_blog_post"],
        research_data=state["research_data"],
        topic=state["topic"]
    )
    
    # Check for hallucinations in the Tier 2 results
    hallucination_found = results["tier2"].get("hallucination_detected", False)
    if hallucination_found:
        print("⚠️  CRITICAL: Hallucination detected by NLI Fact-Check!")

    return {
        "quality_evaluation": results,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def decide_to_finish(state: AgentState):
    eval_results = state.get("quality_evaluation", {})
    final_score = eval_results.get("final_score", 0)
    iterations = state.get("iteration_count", 0)
    
    is_good_enough = final_score >= 7.5
    has_hallucination = eval_results.get("tier2", {}).get("hallucination_detected", False)

    if (is_good_enough and not has_hallucination) or iterations >= 3:
        print(f"--- ✅ DECISION: FINISH (Score: {final_score}) ---")
        return END  # Change "image_gen" to END here
    else:
        print(f"--- 🔄 DECISION: RE-WRITE ---")
        return "writer"

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("evaluator", evaluator_node)
    # workflow.add_node("image_gen", image_generator_node) # Keeping this commented
    
    workflow.set_entry_point("researcher")
    
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "evaluator")
    
    workflow.add_conditional_edges(
        "evaluator",
        decide_to_finish,
        {
            "writer": "writer",
            END: END  # Change "image_gen": "image_gen" to END: END
        }
    )
    
    return workflow.compile()
# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def run_app():
    topic = input("📝 Enter blog topic: ").strip()
    
    # 1. Topic Gatekeeper
    validator = TopicValidator()
    validation = validator.validate(topic)
    if not validation["valid"]:
        print(f"❌ Topic Rejected: {validation['reason']}")
        return

    # 2. Setup State
    initial_state = {
        "topic": topic,
        "iteration_count": 0,
        "error": None,
        "sources": [],
        "research_data": "",
        "blog_outline": "",
        "final_blog_post": "",
        "fact_check_report": ""
    }

    # 3. Run Workflow
    app = build_graph()
    final_output = app.invoke(initial_state)

    # 4. Final Display
    print("\n" + "="*80)
    print("🚀 WORKFLOW COMPLETE")
    print("="*80)
    
    if final_output.get("final_blog_post"):
        print(final_output["final_blog_post"])
        print("\n" + "="*80)
        eval_data = final_output.get("quality_evaluation", {})
        print(f"Final Score: {eval_data.get('final_score')}/10")
        print(f"Factual Status: {final_output.get('quality_evaluation', {}).get('verdict')}")
        print(f"Iterations: {final_output.get('iteration_count')}")
        print(f"Fact Checking : {final_output.get('fact_check_report')}")

    else:
        print(f"Error: {final_output.get('error')}")

if __name__ == "__main__":
    run_app()