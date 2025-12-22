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
    fact_checker_node
)
from validators import TopicValidator, realistic_evaluation

# ---------------------------------------------------------------------------
# NEW NODES FOR THE GRAPH
# ---------------------------------------------------------------------------

def evaluator_node(state: AgentState):
    """Evaluates the blog quality using our math-based validator."""
    print("--- 📊 EVALUATING QUALITY ---")
    
    results = realistic_evaluation(
        blog_post=state["final_blog_post"],
        research_data=state["research_data"],
        topic=state["topic"]
    )
    
    return {
        "quality_evaluation": results,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def decide_to_finish(state: AgentState):
    """Conditional edge: Should we rewrite or finish?"""
    eval_results = state.get("quality_evaluation")
    iterations = state.get("iteration_count", 0)

    # If score is good (>7.5) or we've tried 3 times already, stop.
    if eval_results["final_score"] >= 7.5 or iterations >= 3:
        print("--- ✅ DECISION: FINISH ---")
        return END
    else:
        print(f"--- 🔄 DECISION: RE-WRITE (Score: {eval_results['final_score']}) ---")
        return "writer"

# ---------------------------------------------------------------------------
# GRAPH CONSTRUCTION
# ---------------------------------------------------------------------------

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("evaluator", evaluator_node)
    
    workflow.set_entry_point("researcher")
    
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "evaluator")
    
    # THE LOOP: This is where the magic happens
    workflow.add_conditional_edges(
        "evaluator",
        decide_to_finish,
        {
            "writer": "writer",
            END: END
        }
    )
    
    return workflow.compile()

# ---------------------------------------------------------------------------
# MAIN RUNNER
# ---------------------------------------------------------------------------

def run_app():
    topic = input("📝 Enter blog topic: ").strip()
    
    # Pre-Graph Validation
    validator = TopicValidator()
    valid, reason = validator.validate(topic)["valid"], validator.validate(topic)["reason"]
    
    if not valid:
        print(f"❌ Topic Rejected: {reason}")
        return

    # Initialize State
    initial_state = {
        "topic": topic,
        "iteration_count": 0,
        "error": None,
        "sources": []
    }

    # Run Graph
    app = build_graph()
    final_output = app.invoke(initial_state)

    # Final Display
    if final_output.get("error"):
        print(f"!! Error: {final_output['error']}")
    else:
        print("\n" + "="*50)
        print("🔥 FINAL BLOG POST 🔥")
        print("="*50)
        print(final_output["final_blog_post"])
        print("\n" + "="*50)
        print(f"Final Score: {final_output['quality_evaluation']['final_score']}/10")
        print(f"Iterations: {final_output['iteration_count']}")
        print("="*50)

if __name__ == "__main__":
    run_app()