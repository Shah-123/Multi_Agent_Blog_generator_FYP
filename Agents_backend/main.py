import os
from dotenv import load_dotenv

# 1. Environment Setup
load_dotenv()

from langgraph.graph import StateGraph, END
from Graph.state import AgentState
# Correct Imports
from Graph.nodes import (
    researcher_node, 
    analyst_node, 
    writer_node, 
    fact_checker_node,
    image_generator_node,
    social_media_node 
)
from validators import TopicValidator, realistic_evaluation

# ---------------------------------------------------------------------------
# GRAPH NODES (Evaluator Logic)
# ---------------------------------------------------------------------------

def evaluator_node(state: AgentState):
    """Evaluates the blog quality."""
    print("--- 📊 EVALUATING QUALITY ---")
    
    results = realistic_evaluation(
        blog_post=state["final_blog_post"],
        research_data=state.get("raw_research_data", ""),
        topic=state["topic"]
    )
    
    hallucination = results["tier2"].get("hallucination_detected", False)
    if hallucination:
        print("⚠️  CRITICAL: Hallucination detected!")

    return {
        "quality_evaluation": results,
        "iteration_count": state.get("iteration_count", 0) + 1
    }

def decide_to_finish(state: AgentState):
    eval_results = state.get("quality_evaluation", {})
    final_score = eval_results.get("final_score", 0)
    iterations = state.get("iteration_count", 0)
    hallucination = eval_results.get("tier2", {}).get("hallucination_detected", False)

    if (final_score >= 7.5 and not hallucination) or iterations >= 3:
        print(f"--- ✅ DECISION: FINISH (Score: {final_score}) ---")
        return END  
    else:
        print(f"--- 🔄 DECISION: RE-WRITE (Score: {final_score}) ---")
        return "writer"

def build_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("image_gen", image_generator_node)
    workflow.add_node("social_media", social_media_node)
    workflow.add_node("evaluator", evaluator_node)
    
    # Set Entry Point
    workflow.set_entry_point("researcher")
    
    # Linear Sequence (Safe & Stable)
    workflow.add_edge("researcher", "analyst")
    workflow.add_edge("analyst", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "image_gen") 
    workflow.add_edge("image_gen", "social_media") 
    workflow.add_edge("social_media", "evaluator") 
    
    # Conditional Edge
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
    tone = input("🎭 Enter tone (Professional, Funny): ").strip() or "Professional"
    
    print("\nSelect Plan:")
    print("1. Basic")
    print("2. Premium")
    choice = input("Select (1 or 2): ").strip()
    plan = "premium" if choice == "2" else "basic"
    
    validator = TopicValidator()
    validation = validator.validate(topic)
    if not validation["valid"]:
        print(f"❌ Topic Rejected: {validation['reason']}")
        return

    # Initialize State
    initial_state = {
        "topic": topic,
        "tone": tone,
        "plan": plan, 
        "iteration_count": 0,
        "error": None,
        "sources": [],
        "research_data": None,
        "raw_research_data": "",
        "competitor_headers": "",
        "blog_outline": None,
        "sections": [],
        "seo_metadata": {}, 
        "final_blog_post": "",
        "fact_check_report": None,
        "image_path": "",
        "linkedin_post": "",
        "youtube_script": "",
        "facebook_post": ""
    }

    print(f"🚀 STARTING WORKFLOW (Topic: {topic}, Tone: {tone}, Plan: {plan.upper()})...")
    app = build_graph()
    final_output = app.invoke(initial_state)

    print("\n" + "="*80)
    print("🚀 WORKFLOW COMPLETE")
    print("="*80)
    
    if final_output.get("final_blog_post"):
        print(final_output["final_blog_post"])
        
        print("\n" + "="*80)
        print("📈 SEO STRATEGY")
        print("="*80)
        seo = final_output.get("seo_metadata", {})
        print(f"Title:       {seo.get('title')}")
        print(f"Description: {seo.get('description')}")
        print(f"Keywords:    {seo.get('keywords')}")
        
        print("\n" + "="*80)
        print("🎨 IMAGE RESULT")
        print("="*80)
        img_path = final_output.get("image_path")
        if img_path:
            print(f"✅ Saved at: {os.path.abspath(img_path)}")
        else:
            print("❌ Image generation failed.")

        print("\n" + "="*80)
        print("📱 SOCIAL MEDIA PACK")
        print("="*80)
        print("--- LINKEDIN ---")
        print(final_output.get("linkedin_post", "") + "...\n")
        print("--- LINKEDIN ---")
        
        print(final_output.get('youtube_script', "") + "...\n")
        print("--- LINKEDIN ---")
        
        print(final_output.get('facebook_post', "") + "...\n")

        print("\n" + "="*80)
        eval_data = final_output.get("quality_evaluation", {})
        print(f"Final Score: {eval_data.get('final_score')}/10")
        
    else:
        print(f"Error: {final_output.get('error')}")

if __name__ == "__main__":
    run_app()