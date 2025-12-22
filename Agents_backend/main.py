from dotenv import load_dotenv

# Load environment variables FIRST before importing anything else
load_dotenv()

from langgraph.graph import StateGraph, END
from Graph.state import AgentState
from Graph.nodes import researcher_node, analyst_node, writer_node, fact_checker_node, regenerate_blog_with_feedback
from validators import TopicValidator, LLMFeedback, realistic_evaluation
from langchain_openai import ChatOpenAI
def build_graph():
    """Build and compile the agentic workflow graph with fact-checking."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    
    # Set entry point
    workflow.set_entry_point("researcher")
    
    # Define edges - the flow of data through the graph
    workflow.add_edge("researcher", "analyst")       # Researcher → Analyst
    workflow.add_edge("analyst", "writer")           # Analyst → Writer
    workflow.add_edge("writer", "fact_checker")      # Writer → Fact-Checker
    workflow.add_edge("fact_checker", END)           # Fact-Checker → End
    
    return workflow.compile()





def run_workflow(topic: str):
    """
    Run complete workflow with validation and evaluation.
    Single invoke() - no streaming.
    """
    
    # Initialize validators
        # Initialize validators
    topic_validator = TopicValidator()
    llm_feedback = LLMFeedback()
    
    # Step 1: Validate topic
    print("\n" + "=" * 80)
    print("🔍 VALIDATING TOPIC")
    print("=" * 80)
    
    validation_result = topic_validator.validate(topic)
    
    if not validation_result["valid"]:
        print(f"\n❌ TOPIC REJECTED: {validation_result['reason']}")
        print(f"   Severity: {validation_result.get('severity', 'UNKNOWN')}")
        return
    
    # Step 2: Build graph ONCE
    print("\n" + "=" * 80)
    print("🚀 MULTI-AGENT BLOG GENERATION WORKFLOW")
    print("=" * 80)
    print(f"📌 Topic: {topic}")
    print("=" * 80 + "\n")
    
    app = build_graph()
    
    # Step 3: Initialize state with ALL required fields
    inputs = {
        "topic": topic.strip(),
        "research_data": "",
        "sources": [],
        "blog_outline": "",
        "final_blog_post": "",
        "fact_check_report": "",
        "error": None,
    }
    
    # Step 4: RUN WORKFLOW - SINGLE invoke() call
    print("⏳ Running workflow... (this may take 1-2 minutes)\n")
    final_state = app.invoke(inputs)
    
    # Step 5: Check for errors
    if final_state.get("error"):
        print("\n" + "=" * 80)
        print("❌ WORKFLOW FAILED")
        print("=" * 80)
        print(f"\n❌ ERROR: {final_state['error']}")
        return
    
    # Step 6: Display results
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Show researcher results
    if final_state.get("research_data"):
        print("\n📊 RESEARCH DATA GENERATED:\n")
        sources = final_state.get("sources", [])
        if sources:
            print(f"✓ Found {len(sources)} Sources:\n")
            for i, source in enumerate(sources, 1):
                print(f"{i}. {source.get('title', 'Untitled')}")
                print(f"   URL: {source.get('url', 'N/A')}\n")
        
        # Show research preview
        research_preview = final_state["research_data"][:400] if len(final_state["research_data"]) > 400 else final_state["research_data"]
        print(f"📄 Research Summary:\n{research_preview}\n")
    
    # Show analyst results
    if final_state.get("blog_outline"):
        print("\n" + "=" * 80)
        print("📋 BLOG OUTLINE GENERATED")
        print("=" * 80 + "\n")
        outline_preview = final_state["blog_outline"][:800] if len(final_state["blog_outline"]) > 800 else final_state["blog_outline"]
        print(outline_preview + "\n")
    
    # Show FULL writer results (THIS WAS THE BUG!)
    if final_state.get("final_blog_post"):
        print("\n" + "=" * 80)
        print("✍️ FINAL BLOG POST GENERATED")
        print("=" * 80)
        blog = final_state.get("final_blog_post", "")
        print(f"📏 Blog Length: {len(blog)} characters\n")
        
        # SHOW THE FULL BLOG (not just preview!)
        print(blog)
        print("\n")
    
    # Show fact-checker results
    if final_state.get("fact_check_report"):
        print("\n" + "=" * 80)
        print("🔍 FACT-CHECK REPORT")
        print("=" * 80 + "\n")
        print(final_state["fact_check_report"])
        print("\n")
    
    # Step 7: Evaluate quality and get LLM feedback
    print("\n" + "=" * 80)
    print("📊 EVALUATING BLOG QUALITY")
    print("=" * 80 + "\n")
    
    blog_post = final_state.get("final_blog_post", "")
    research_data = final_state.get("research_data", "")
    
    quality_details = realistic_evaluation(
        blog_post=blog_post,
        research_data=research_data,
        topic=topic
    )
    
    # Store evaluation in state
    final_state["quality_evaluation"] = quality_details
    
    # Step 8: AUTO-REGENERATE BLOG BASED ON FEEDBACK
    print("\n" + "=" * 80)
    print("🔄 AUTO-REGENERATING BLOG WITH LLM FEEDBACK")
    print("=" * 80 + "\n")
    
    llm_feedback_obj = quality_details.get("tier3", {}).get("feedback", "")
    
    if llm_feedback_obj:
        # Regenerate the blog automatically
        improved_blog = regenerate_blog_with_feedback(
            blog_outline=final_state.get("blog_outline", ""),
            research_data=research_data,
            topic=topic,
            llm_feedback=llm_feedback_obj,
            iteration=1
        )
        
        # Update the state with improved blog
        final_state["final_blog_post"] = improved_blog
        
        print("\n" + "=" * 80)
        print("✍️ IMPROVED BLOG POST")
        print("=" * 80)
        print(f"📏 New Blog Length: {len(improved_blog)} characters\n")
        print(improved_blog)
        print("\n")
        
        # Optional: Run fact-check on improved blog
        print("\n" + "=" * 80)
        print("🔍 FACT-CHECKING IMPROVED BLOG")
        print("=" * 80 + "\n")
        
        from Graph.nodes import fact_checker_node
        
        fact_check_state = {
            "topic": topic,
            "research_data": research_data,
            "final_blog_post": improved_blog,
            "error": None
        }
        
        fact_check_result = fact_checker_node(fact_check_state)
        improved_fact_check = fact_check_result.get("fact_check_report", "No issues found")
        
        print(improved_fact_check)
        print("\n")
        
        final_state["fact_check_report"] = improved_fact_check
    else:
        print("⚠️  No feedback available for regeneration")
    
    print("\n" + "=" * 80)
    print("✅ ALL TASKS COMPLETED")
    print("=" * 80 + "\n")
    
    return final_state


if __name__ == "__main__":
    topic = input("📝 Enter a topic for blog generation: ").strip()
    run_workflow(topic)