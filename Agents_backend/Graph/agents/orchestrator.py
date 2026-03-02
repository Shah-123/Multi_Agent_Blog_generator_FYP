from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State, Plan
from Graph.templates import ORCH_SYSTEM
from .utils import logger, llm, _job, _emit

# ✅ FIX #7: Replace the confusing 4-variable arithmetic chain with named constants
# that clearly document the blog structure.
#
# BEFORE (what it was):
#   target_sections_math     = target_sections + 2
#   target_sections_plus_one = target_sections_math + 1
#   target_sections_plus_two = target_sections_plus_one + 1
#   total_sections           = target_sections_plus_two  # = target_sections + 4
#
# All four variables were just computing target_sections + 4, but the names
# made it impossible to tell why. If the structure ever changed (e.g. adding
# a new fixed section), you had to trace through all four to understand the impact.
#
# AFTER — the structure is self-documenting:
#
#   FIXED_OPENING_SECTIONS  = 2  (Intro + Context)
#   FIXED_CLOSING_SECTIONS  = 2  (Practical Application + Actionable Takeaways)
#   TOTAL_FIXED_SECTIONS    = 4
#
# To add or remove a fixed section in future, change ONE constant.

FIXED_OPENING_SECTIONS = 2   # Section 1: Hook/Intro, Section 2: Context/Background
FIXED_CLOSING_SECTIONS = 2   # Second-to-last: Practical Application, Last: Actionable Takeaways
TOTAL_FIXED_SECTIONS   = FIXED_OPENING_SECTIONS + FIXED_CLOSING_SECTIONS  # = 4


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

    # ✅ FIX #7: Clear, named section breakdown passed to the prompt template.
    #
    # Blog structure:
    #   [1]              = Hook / Intro
    #   [2]              = Context / Background
    #   [3 .. N+2]       = Body sections  (N = target_sections, user-controlled)
    #   [N+3]            = Practical Application
    #   [N+4]            = Actionable Takeaways
    #
    # Example: user picks 5 body sections → total = 5 + 4 = 9 sections
    total_sections           = target_sections + TOTAL_FIXED_SECTIONS
    last_body_section        = target_sections + FIXED_OPENING_SECTIONS
    practical_section_number = last_body_section + 1
    takeaways_section_number = last_body_section + 2

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM.format(
            tone=target_tone,
            keywords=keywords_str,
            target_sections=last_body_section,
            target_sections_plus_one=practical_section_number,
            target_sections_plus_two=takeaways_section_number,
            total_sections=total_sections,
        )),
        HumanMessage(content=prompt_content),
    ])

    logger.info(f"Generated {len(plan.tasks)} sections.")
    logger.info(f"🎨 Tone: {plan.tone}")
    if plan.primary_keywords:
        logger.info(f"🎯 Keywords: {', '.join(plan.primary_keywords)}")

    _emit(_job(state), "orchestrator", "completed", f"Planned {len(plan.tasks)} sections", {"sections": len(plan.tasks), "tone": plan.tone})
    return {"plan": plan}