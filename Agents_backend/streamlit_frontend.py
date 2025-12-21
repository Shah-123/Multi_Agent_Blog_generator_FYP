import streamlit as st
import requests
import json
import time

# Configuration
API_URL = "http://127.0.0.1:8000/generate-blog"

st.set_page_config(page_title="AI Blog Agent", layout="wide")

st.title("🤖 AI Multi-Agent Blog Generator")
st.markdown("### Research • Write • Fact-Check")

# Input Section
with st.sidebar:
    st.header("Configuration")
    topic = st.text_input("Enter Blog Topic:", placeholder="e.g., History of Pakistan")
    generate_btn = st.button("Generate Blog", type="primary")
    st.info("This pipeline uses Tavily for research and GPT-4o for writing and verification.")

# Main Logic
if generate_btn and topic:
    # Create placeholders for streaming
    status_container = st.container()
    progress_bar = st.progress(0)
    
    # Progress tracking
    progress_steps = {
        "Researcher": 0,
        "Analyst": 25,
        "Writer": 50,
        "Fact Checker": 75,
        "Complete": 100
    }
    current_progress = 0
    
    try:
        # Stream from API
        response = requests.post(API_URL, json={"topic": topic}, stream=True)
        
        if response.status_code == 200:
            final_data = None
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    event = json.loads(line)
                    
                    if event["status"] == "starting":
                        with status_container:
                            st.info("⏳ " + event["message"])
                    
                    elif event["status"] == "progress":
                        node = event.get("node")
                        current_progress = progress_steps.get(node, 0)
                        
                        with status_container:
                            st.info(f"{current_progress//25 + 1}/5: {event['message']}")
                        
                        progress_bar.progress(current_progress / 100)
                        time.sleep(0.5)  # Visual feedback
                    
                    elif event["status"] == "complete":
                        final_data = event.get("data")
                        progress_bar.progress(1.0)
                        st.success("✅ Blog generation complete!")
            
            if final_data:
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Blog", "🔍 Fact Check", "📊 Research", "📋 Outline"])
                
                # 1. Final Blog Post
                with tab1:
                    st.subheader(final_data["topic"].title())
                    st.markdown(final_data.get("final_blog_post", "No blog post generated"))
                
                # 2. Fact Check Report
                with tab2:
                    st.warning("Always verify AI content before publishing.")
                    st.markdown(final_data.get("fact_check_report", "No fact check report"))
                
                # 3. Research Data & Sources
                with tab3:
                    st.subheader("Sources Used")
                    sources = final_data.get("sources", [])
                    if sources:
                        for source in sources:
                            st.markdown(f"- [{source.get('title', 'Link')}]({source['url']})")
                    else:
                        st.info("No sources found")
                    
                    st.divider()
                    st.subheader("Raw Research Data")
                    with st.expander("View Full Research Summary"):
                        st.markdown(final_data.get("research_data", "No research data"))
                
                # 4. Outline
                with tab4:
                    st.markdown(final_data.get("blog_outline", "No outline generated"))
        
        else:
            st.error(f"Server Error: {response.status_code}")
    
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to the Backend API. Is 'uvicorn Api:app' running?")
    except Exception as e:
        st.error(f"Error: {str(e)}")