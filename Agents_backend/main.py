from dotenv import load_dotenv

# Load environment variables FIRST before importing anything else
load_dotenv()

from langgraph.graph import StateGraph, END
from Graph.state import AgentState
from Graph.nodes import researcher_node, analyst_node, writer_node

def build_graph():
    """Build and compile the agentic workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("writer", writer_node)
    
    # Set entry point
    workflow.set_entry_point("researcher")
    
    # Define edges - the flow of data through the graph
    workflow.add_edge("researcher", "analyst")   # Researcher passes data to Analyst
    workflow.add_edge("analyst", "writer")       # Analyst passes outline to Writer
    workflow.add_edge("writer", END)             # Writer finishes the workflow
    
    return workflow.compile()

def run_workflow(topic: str):
    """
    Run the complete workflow with input validation and formatted output.
    
    Args:
        topic: The research topic
    """
    if not topic or not topic.strip():
        print("Error: Please provide a valid topic.")
        return
    
    app = build_graph()
    inputs = {"topic": topic.strip()}
    
    print("\n" + "=" * 70)
    print(f"🚀 STARTING MULTI-AGENT BLOG GENERATION WORKFLOW")
    print(f"📌 TOPIC: {topic}")
    print("=" * 70 + "\n")
    
    for output in app.stream(inputs):
        for node_name, state in output.items():
            print(f"\n✓ {node_name.upper()} NODE COMPLETED")
            print("-" * 70)
            
            if state.get("error"):
                print(f"❌ ERROR: {state['error']}")
            elif node_name == "researcher" and state.get("research_data"):
                print("\n📊 RESEARCH DATA GENERATED:\n")
                # Print first 500 chars as preview
                data_preview = state["research_data"][:500] + "..." if len(state["research_data"]) > 500 else state["research_data"]
                print(data_preview)
            elif node_name == "analyst" and state.get("blog_outline"):
                print("\n📋 BLOG OUTLINE GENERATED:\n")
                print(state["blog_outline"])
            elif node_name == "writer" and state.get("final_blog_post"):
                print("\n✍️ FINAL BLOG POST GENERATED:\n")
                print(state["final_blog_post"])
            
            print("-" * 70)
    
    print("\n" + "=" * 70)
    print("✅ WORKFLOW COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    topic = input("Enter a topic for blog generation: ").strip()
    run_workflow(topic)