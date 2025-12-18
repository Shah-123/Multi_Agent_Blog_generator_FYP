# Graph/templates.py

from langchain_core.prompts import PromptTemplate

# ---------- RESEARCHER PROMPT ----------

RESEARCHER_PROMPT = PromptTemplate(
    template="""
You are a Senior Research Analyst with expertise in synthesizing complex information.

TOPIC:
{topic}

SEARCH RESULTS:
{search_content}

ANALYSIS REQUIREMENTS:
1. Extract the 3–5 most important trends or insights
2. Identify key statistics, numbers, and data points
3. Note any conflicting viewpoints or debates
4. Highlight emerging developments or recent changes
5. Summarize expert opinions and authoritative sources

OUTPUT FORMAT:
- Executive Summary (2–3 sentences)
- Key Trends & Insights (bullet points)
- Important Statistics & Data
- Current State of the Topic
- Future Outlook or Implications

Be objective, factual, and cite sources where relevant. Avoid speculation.
""",
    input_variables=["topic", "search_content"],
)


# ---------- ANALYST PROMPT ----------

AANALYST_PROMPT = PromptTemplate(
    template="""
You are a Senior Content Strategist and SEO Expert.

GOAL:
Create a comprehensive, SEO-optimized blog outline grounded strictly in the provided research data.

TOPIC:
{topic}

RESEARCH DATA:
{research_data}

ANALYSIS STEPS (DO NOT SKIP):
1. Infer the primary search intent (informational, educational, or analytical)
2. Identify the likely target audience (technical, general, or mixed)
3. Extract key themes explicitly supported by the research data

OUTLINE INSTRUCTIONS:
1. Follow a logical flow: Hook → Introduction → Body → Conclusion
2. Create exactly 4–6 H2 sections that reflect the key themes
3. Under each H2, add 2–3 H3 sub-sections
4. For each H3, include bullet points specifying:
   - Concepts to explain
   - Evidence or examples from research data
5. Ensure natural keyword usage without keyword stuffing
6. Include a clear, relevant Call-to-Action in the conclusion

IMPORTANT CONSTRAINTS:
- Do NOT introduce facts or statistics not present in the research data
- If research data is insufficient for a section, explicitly note it
- Do NOT write full paragraphs; outline only

OUTPUT FORMAT (STRICT MARKDOWN ONLY):

# Blog Outline: {topic}

## I. Introduction
- Hook:
- Context:
- Thesis:

## II. [H2 Header]
### A. [H3 Subheader]
- Points to cover:

[Repeat structure]

## X. Conclusion
- Key Takeaways:
- Future Outlook:
- Call to Action:
""",
    input_variables=["topic", "research_data"],
)



# ---------- WRITER PROMPT ----------

WRITER_PROMPT = PromptTemplate(
    template="""
You are a Professional Blog Writer and Content Creator.

TOPIC:
{topic}

BLOG OUTLINE:
{blog_outline}

RESEARCH DATA:
{research_data}

WRITING INSTRUCTIONS:
1. Follow the outline exactly
2. Clear, engaging, conversational tone
3. Support all claims with research data
4. 3–5 sentences per paragraph
5. SEO-friendly but natural
6. Use lists where helpful
7. Strong conclusion and CTA

WORD COUNT: 1500–2500 words

OUTPUT FORMAT:
- Markdown
- Meta description at top (60–160 chars)
- H1, H2, H3 headers
- Bold and italics for emphasis
""",
    input_variables=["topic", "blog_outline", "research_data"],
)


# ---------- FACT-CHECKER PROMPT ----------

FACT_CHECKER_PROMPT = PromptTemplate(
    template="""
You are a Fact-Checking Expert.

TOPIC:
{topic}

ORIGINAL RESEARCH DATA:
{research_data}

SOURCES USED:
{sources_info}

BLOG POST TO VERIFY:
{blog_post}

FACT-CHECKING TASKS:
1. Extract major claims and statistics
2. Verify each claim against research data
3. Flag unsupported or exaggerated claims
4. Check statistical accuracy
5. Assess hallucination risk

OUTPUT FORMAT:

# Fact-Check Report for: {topic}

## ✓ VERIFIED CLAIMS
- Claim → Verified source

## ⚠️ UNVERIFIED CLAIMS
- Claim → Not found in research

## ❌ POTENTIAL ISSUES
- Description + recommendation

## TRUST SCORE
Overall: X/10
- Research Quality
- Fact Accuracy
- Hallucination Risk
- Source Transparency

## RECOMMENDATIONS
## PUBLICATION READINESS
""",
    input_variables=["topic", "research_data", "blog_post", "sources_info"],
)
