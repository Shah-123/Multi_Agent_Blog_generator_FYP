import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import LangChainException
from Graph.state import AgentState

# Setup Tools and LLM
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

tavily_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_api_key)
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# --- NODE 1: RESEARCHER ---
def researcher_node(state: AgentState) -> dict:
    """
    Research node that searches for information and structures findings.
    
    Args:
        state: Current agent state containing topic
        
    Returns:
        Updated state with research_data or error message
    """
    try:
        topic = state['topic']
        
        if not topic or not topic.strip():
            return {"error": "Topic cannot be empty"}
        
        print(f"--- RESEARCHER AGENT WORKING ---")
        print(f"Searching for: {topic}")
        
        # 1. Search the internet
        search_results = tavily_tool.invoke(topic)
        
        if not search_results:
            return {"error": "No search results found for the topic"}
        
        # 2. Format search results
        search_content = "\n".join([
            f"Source: {result['url']}\nContent: {result['content']}"
            for result in search_results
        ])
        
        # 3. Summarize using LLM with improved prompt
        prompt = PromptTemplate(
            template="""You are a Senior Research Analyst with expertise in synthesizing complex information. Your task is to analyze and structure search results into a comprehensive research report.

TOPIC: {topic}

SEARCH RESULTS:
{search_content}

ANALYSIS REQUIREMENTS:
1. Extract the 3-5 most important trends or insights
2. Identify key statistics, numbers, and data points
3. Note any conflicting viewpoints or debates
4. Highlight emerging developments or recent changes
5. Summarize expert opinions and authoritative sources

OUTPUT FORMAT:
Provide a structured report with clear sections:
- Executive Summary (2-3 sentences)
- Key Trends & Insights (bullet points)
- Important Statistics & Data
- Current State of the Topic
- Future Outlook or Implications

Be objective, factual, and cite sources where relevant. Avoid speculation.""",
            input_variables=["topic", "search_content"]
        )
        
        response = prompt | llm
        result = response.invoke({"topic": topic, "search_content": search_content})
        
        return {"research_data": result.content}
        
    except LangChainException as e:
        return {"error": f"Search error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# --- NODE 2: ANALYST (CONTENT STRATEGIST) ---
def analyst_node(state: AgentState) -> dict:
    """
    Analyst node that transforms research data into a structured, SEO-optimized blog outline.
    
    Args:
        state: Current agent state containing topic and research_data
        
    Returns:
        Updated state with blog_outline or error message
    """
    try:
        research_data = state.get('research_data')
        topic = state.get('topic')
        
        # Defensive coding: ensure we have data
        if state.get('error'):
            return {"error": f"Cannot analyze: {state.get('error')}"}
        
        if not research_data:
            return {"error": "No research data available to generate outline"}
        
        print(f"--- ANALYST AGENT WORKING ---")
        print(f"Developing content strategy for: {topic}")
        
        prompt = PromptTemplate(
            template="""You are a Senior Content Strategist and SEO Expert.

GOAL: Create a comprehensive, SEO-optimized blog post outline based on research data.

TOPIC: {topic}

RESEARCH DATA:
{research_data}

INSTRUCTIONS:
1. Identify the target audience based on topic complexity and search intent
2. Create a logical flow: Hook → Introduction → Body Sections → Conclusion
3. Define 4-6 main H2 headers that cover different angles of the topic
4. Under each H2, add 2-3 H3 subheaders with specific content points
5. Include brief bullet points under each section describing what to cover
6. Optimize for SEO by including natural keyword placement opportunities
7. Ensure the structure is scannable and reader-friendly
8. Add a compelling Call-to-Action section at the end

OUTPUT FORMAT (Strict Markdown):

# Blog Outline: {topic}

## I. Introduction
- Hook: [Brief compelling opening statement]
- Problem Statement: [What problem does this topic solve?]
- Thesis: [Main argument or value proposition]

## II. [H2 Header 1]
### A. [H3 Subheader]
- Point to cover
- Point to cover

### B. [H3 Subheader]
- Point to cover

## III. [H2 Header 2]
### A. [H3 Subheader]
- Point to cover

[Continue with remaining sections...]

## X. Conclusion
- Key Takeaways: [Summarize main points]
- Future Implications: [What's next?]
- Call to Action: [How should readers engage?]

IMPORTANT: Make the outline specific to the research data provided. Use actual statistics, trends, and insights from the research.""",
            input_variables=["topic", "research_data"]
        )
        
        chain = prompt | llm
        response = chain.invoke({"topic": topic, "research_data": research_data})
        
        return {"blog_outline": response.content}
        
    except LangChainException as e:
        return {"error": f"Analyst error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error in analyst: {str(e)}"}

# --- NODE 3: WRITER (CONTENT CREATOR) ---
def writer_node(state: AgentState) -> dict:
    """
    Writer node that transforms the blog outline into a complete, polished blog post.
    Uses the outline as skeleton and research data as reference material.
    
    Args:
        state: Current agent state containing topic, research_data, and blog_outline
        
    Returns:
        Updated state with final_blog_post or error message
    """
    try:
        topic = state.get('topic')
        research_data = state.get('research_data')
        blog_outline = state.get('blog_outline')
        
        # Defensive coding: ensure we have required data
        if state.get('error'):
            return {"error": f"Cannot write: {state.get('error')}"}
        
        if not blog_outline:
            return {"error": "No blog outline available to write from"}
        
        if not research_data:
            return {"error": "No research data available for reference"}
        
        print(f"--- WRITER AGENT WORKING ---")
        print(f"Writing blog post for: {topic}")
        
        prompt = PromptTemplate(
            template="""You are a Professional Blog Writer and Content Creator with expertise in engaging, informative writing.

GOAL: Write a complete, high-quality blog post based on the provided outline and research data.

TOPIC: {topic}

BLOG OUTLINE (STRUCTURE TO FOLLOW):
{blog_outline}

RESEARCH DATA (FACTS & REFERENCE):
{research_data}

WRITING INSTRUCTIONS:
1. Follow the outline structure exactly - use the headers and sections provided
2. Write in a clear, engaging, and conversational tone (not robotic)
3. For each section, develop the ideas with:
   - Clear explanations and context
   - Real data and statistics from the research
   - Practical examples where applicable
   - Smooth transitions between ideas
4. Make the introduction compelling with a strong hook that captures attention
5. Use the research data to back up every major claim
6. Keep paragraphs concise (3-5 sentences max) for better readability
7. Use bullet points or numbered lists when appropriate for clarity
8. Include relevant keywords naturally throughout the post (no keyword stuffing)
9. End with a strong conclusion that summarizes key takeaways
10. Include a compelling Call-to-Action that encourages reader engagement
11. Maintain consistent tone and voice throughout
12. Use formatting strategically: bold for emphasis, italics for definitions, etc.

WORD COUNT TARGET: 1500-2500 words

OUTPUT FORMAT:
Write the complete blog post in Markdown format. Include:
- # Main Title (H1)
- ## Section Headers (H2)
- ### Subsection Headers (H3)
- Natural formatting for readability
- Meta description suggestion at the top (60-160 characters)

QUALITY REQUIREMENTS:
- Well-researched and factually accurate
- Engaging and easy to read
- SEO-friendly
- Professional yet approachable tone
- Actionable insights or takeaways""",
            input_variables=["topic", "blog_outline", "research_data"]
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "topic": topic,
            "blog_outline": blog_outline,
            "research_data": research_data
        })
        
        return {"final_blog_post": response.content}
        
    except LangChainException as e:
        return {"error": f"Writer error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error in writer: {str(e)}"}