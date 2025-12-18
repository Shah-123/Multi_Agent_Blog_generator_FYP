import streamlit as st
import requests
import json

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
    with st.spinner(f"Agents are working on '{topic}'... This may take a minute."):
        try:
            # Call the FastAPI Backend
            response = requests.post(API_URL, json={"topic": topic})
            
            if response.status_code == 200:
                data = response.json()
                
                # Check for functional errors inside the JSON
                if data.get("error"):
                    st.error(f"Agent Error: {data['error']}")
                else:
                    # TABS LAYOUT
                    tab1, tab2, tab3, tab4 = st.tabs(["📄 Final Blog", "🔍 Fact Check", "📊 Research", "📋 Outline"])
                    
                    # 1. Final Blog Post
                    with tab1:
                        st.subheader(data["topic"].title())
                        st.markdown(data["final_blog_post"])
                    
                    # 2. Fact Check Report
                    with tab2:
                        st.warning("Always verify AI content before publishing.")
                        st.markdown(data["fact_check_report"])
                    
                    # 3. Research Data & Sources
                    with tab3:
                        st.subheader("Sources Used")
                        for source in data["sources"]:
                            st.markdown(f"- [{source.get('title', 'Link')}]({source['url']})")
                        
                        st.divider()
                        st.subheader("Raw Research Data")
                        with st.expander("View Full Research Summary"):
                            st.markdown(data["research_data"])
                            
                    # 4. Outline
                    with tab4:
                        st.markdown(data["blog_outline"])
                        
            else:
                st.error(f"Server Error: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to the Backend API. Is 'uvicorn Api:app' running?")