from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State, Plan
from Graph.templates import ORCH_SYSTEM
from .utils import logger, llm, _job, _emit

def orchestrator_node(state: State) -> dict:
    _emit(_job(state), "orchestrator", "started", "Creating detailed blog outline...")
    logger.info("📋 PLANNING ---")
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])
    
    target_tone = state.get("target_tone", "professional")
    target_keywords = state.get("target_keywords", [])
    
    keywords_str = ", ".join(target_keywords) if target_keywords else "None specified"
    
    prompt_content = f"""Topic: {state['topic']}
Mode: {mode}
Target Tone: {target_tone}
Target Keywords: {keywords_str}

Evidence Context:
{[e.model_dump() for e in evidence][:10]}

Create a blog plan that:
1. Maintains '{target_tone}' tone consistently throughout all sections
2. Naturally integrates these keywords: {keywords_str}
3. Distributes keywords strategically across sections (avoid stuffing)
4. Creates engaging, SEO-optimized content
"""
    
    target_sections = state.get("target_sections", 5)
    
    # Calculate the exact section numbers for the prompt structure
    target_sections_math = target_sections + 2 # Add 2 for Intro and Context
    target_sections_plus_one = target_sections_math + 1
    target_sections_plus_two = target_sections_plus_one + 1
    total_sections = target_sections_plus_two
    
    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM.format(
            tone=target_tone, 
            keywords=keywords_str,
            target_sections=target_sections_math,
            target_sections_plus_one=target_sections_plus_one,
            target_sections_plus_two=target_sections_plus_two,
            total_sections=total_sections
        )),
        HumanMessage(content=prompt_content),
    ])
    
    logger.info(f"Generated {len(plan.tasks)} sections.")
    logger.info(f"🎨 Tone: {plan.tone}")
    if plan.primary_keywords:
        logger.info(f"🎯 Keywords: {', '.join(plan.primary_keywords)}")
    
    _emit(_job(state), "orchestrator", "completed", f"Planned {len(plan.tasks)} sections", {"sections": len(plan.tasks), "tone": plan.tone})
    return {"plan": plan}
