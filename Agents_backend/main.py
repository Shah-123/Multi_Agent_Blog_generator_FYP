from dotenv import load_dotenv

# Load environment variables FIRST before importing anything else
load_dotenv()

from langgraph.graph import StateGraph, END
from Graph.state import AgentState
from Graph.nodes import researcher_node, analyst_node, writer_node, fact_checker_node

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
    Run the complete workflow with input validation and formatted output.
    Includes research, analysis, writing, and fact-checking stages.
    
    Args:
        topic: The research topic
    """
    if not topic or not topic.strip():
        print("❌ Error: Please provide a valid topic.")
        return
    
    app = build_graph()
    inputs = {"topic": topic.strip()}
    
    print("\n" + "=" * 80)
    print("🚀 MULTI-AGENT BLOG GENERATION WORKFLOW")
    print("=" * 80)
    print(f"📌 Topic: {topic}")
    print(f"📊 Stages: Research → Analysis → Writing → Fact-Checking")
    print("=" * 80 + "\n")
    
    for output in app.stream(inputs):
        for node_name, state in output.items():
            print(f"\n{'='*80}")
            print(f"✓ {node_name.upper()} NODE COMPLETED")
            print(f"{'='*80}")
            
            if state.get("error"):
                print(f"\n❌ ERROR: {state['error']}")
            
            elif node_name == "researcher" and state.get("research_data"):
                print("\n📊 RESEARCH DATA GENERATED:\n")
                print("-" * 80)
                
                # Show sources found
                sources = state.get("sources", [])
                if sources:
                    print(f"\n✓ Found {len(sources)} Sources:\n")
                    for i, source in enumerate(sources, 1):
                        print(f"{i}. {source.get('title', 'Untitled')}")
                        print(f"   URL: {source.get('url', 'N/A')}\n")
                
                # Show research data preview
                data_preview = state["research_data"][:600] + "..." if len(state["research_data"]) > 600 else state["research_data"]
                print(f"\n📄 Research Summary:\n{data_preview}\n")
                print("-" * 80)
            
            elif node_name == "analyst" and state.get("blog_outline"):
                print("\n📋 BLOG OUTLINE GENERATED:\n")
                print("-" * 80)
                print(state["blog_outline"])
                print("-" * 80)
            
            elif node_name == "writer" and state.get("final_blog_post"):
                print("\n✍️ FINAL BLOG POST GENERATED:\n")
                print("-" * 80)
                print(state["final_blog_post"])
                print("-" * 80)
            
            elif node_name == "fact_checker" and state.get("fact_check_report"):
                print("\n🔍 FACT-CHECK REPORT:\n")
                print("-" * 80)
                print(state["fact_check_report"])
                print("-" * 80)
    
    print("\n" + "=" * 80)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\n💡 TIP: Review the fact-check report to ensure authenticity before publishing.\n")

if __name__ == "__main__":
    topic = input("📝 Enter a topic for blog generation: ").strip()
    run_workflow(topic)