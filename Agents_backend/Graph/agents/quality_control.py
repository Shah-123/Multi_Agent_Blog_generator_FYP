from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State
from Graph.structured_data import FactCheckReport
from Graph.templates import FACT_CHECKER_SYSTEM, REVISION_SYSTEM
from .utils import logger, llm_quality, _job, _emit

def fact_checker_node(state: State) -> dict:
    _emit(_job(state), "fact_checker", "started", "Auditing claims for accuracy...")
    logger.info("🕵️ FACT CHECKING ---")
    checker = llm_quality.with_structured_output(FactCheckReport)
    
    evidence_summary = "\n".join([
        f"- {e.title} ({e.url})\n  Content: {e.snippet[:500]}..."
        for e in state.get("evidence", [])[:15]
    ])
    
    # If final is not populated yet (because we moved fact checker before reducer), combine the sections
    content_to_audit = state.get("final", "")
    if not content_to_audit:
        sections = state.get("sections", [])
        sections.sort(key=lambda x: x.get("id", 0))
        content_to_audit = "\n\n".join([s.get("content", "") for s in sections])

    report = checker.invoke([
        SystemMessage(content=FACT_CHECKER_SYSTEM),
        HumanMessage(content=(
            f"BLOG CONTENT TO AUDIT:\n{content_to_audit[:8000]}\n\n"
            f"EVIDENCE USED IN RESEARCH:\n{evidence_summary}"
        ))
    ])
    
    report_text = f"FACT CHECK REPORT\n"
    report_text += "=" * 60 + "\n"
    report_text += f"Score: {report.score}/10\n"
    report_text += f"Verdict: {report.verdict}\n\n"
    
    if report.issues:
        report_text += f"Issues Found: {len(report.issues)}\n\n"
        report_text += "DETAILS:\n"
        for i, issue in enumerate(report.issues, 1):
            report_text += f"{i}. [{issue.issue_type}] {issue.claim}\n"
            report_text += f"   -> Fix: {issue.recommendation}\n\n"
    else:
        report_text += "✅ No issues found!\n"
    
    logger.info(f"📊 Score: {report.score}/10 | Verdict: {report.verdict}")
    _emit(_job(state), "fact_checker", "completed", f"Score: {report.score}/10 — {report.verdict}", {"score": report.score, "verdict": report.verdict, "issues": len(report.issues) if report.issues else 0})
    
    issues_list = [
        {
            "claim": issue.claim,
            "issue_type": issue.issue_type,
            "severity": issue.severity,
            "recommendation": issue.recommendation,
        }
        for issue in report.issues
    ] if report.issues else []
    
    return {
        "fact_check_report": report_text,
        "fact_check_verdict": report.verdict,
        "fact_check_issues": issues_list,
        "fact_check_score": report.score,
    }

def revision_node(state: State) -> dict:
    attempts = state.get("fact_check_attempts", 0)
    _emit(_job(state), "revision", "started", f"Self-healing revision (attempt {attempts + 1})...")
    logger.info(f"🔧 REVISING CONTENT (Attempt {attempts + 1})")
    
    issues = state.get("fact_check_issues", [])
    if not issues:
        logger.warning("No issues to fix, skipping revision.")
        return {}
    
    issues_text = "\n".join([
        f"{i+1}. [{iss['issue_type']}] \"{iss['claim']}\"\n   Fix: {iss['recommendation']}"
        for i, iss in enumerate(issues)
    ])
    
    evidence = state.get("evidence", [])
    evidence_text = "\n".join([
        f"- [{e.title}]({e.url})\n  Content: {e.snippet[:400]}"
        for e in evidence[:10]
    ])
    
    # Combine sections if final doesn't exist
    content_to_revise = state.get("final", "")
    sections_existed = False
    
    if not content_to_revise:
        sections = state.get("sections", [])
        sections.sort(key=lambda x: x.get("id", 0))
        content_to_revise = "\n\n".join([s.get("content", "") for s in sections])
        sections_existed = len(sections) > 0
        
    response = llm_quality.invoke([
        SystemMessage(content=REVISION_SYSTEM),
        HumanMessage(content=(
            f"ORIGINAL BLOG:\n{content_to_revise}\n\n"
            f"FLAGGED ISSUES ({len(issues)} total):\n{issues_text}\n\n"
            f"AVAILABLE EVIDENCE (use for citations):\n{evidence_text}"
        )),
    ], max_tokens=8000)
    
    revised = response.content.strip()
    
    original_words = len(content_to_revise.split())
    revised_words = len(revised.split())
    
    if revised_words < (original_words * 0.7):
        logger.warning(f"Revision too short ({revised_words} vs {original_words} words), keeping original.")
        return {"fact_check_attempts": attempts + 1}
    
    logger.info(f"✅ Revised: {revised_words} words (was {original_words})")
    _emit(_job(state), "revision", "completed", f"Revised: {revised_words} words", {"words": revised_words})
    
    # If we are pre-reducer, we must update the state such that the reducer gets the new text. 
    # The reducer loops through `sections` to build `final`.
    # the simplest way to mock the sections is to put the whole revised text into section 0, 
    # and clear the rest, so the reducer just grabs it.
    if sections_existed:
        revised_sections = [{"id": 0, "title": "Revised Blog", "content": revised}]
        return {
            "sections": revised_sections, # Overwrite sections with the revised monolithic blog
            "fact_check_attempts": attempts + 1,
        }
    else:
        return {
            "final": revised,
            "fact_check_attempts": attempts + 1,
        }

def evaluator_node(state: State) -> dict:
    _emit(_job(state), "evaluator", "started", "Scoring final blog quality...")
    logger.info("📊 EVALUATING QUALITY ---")
    try:
        from validators import BlogEvaluator
        evaluator = BlogEvaluator()
        results = evaluator.evaluate(blog_post=state["final"], topic=state["topic"])
        score = results.get('final_score', 0)
        logger.info(f"🏆 Final Score: {score}/10")
        _emit(_job(state), "evaluator", "completed", f"Quality Score: {score}/10", {"score": score})
        return {"quality_evaluation": results}
    except ImportError:
        logger.warning("Validators module not found, skipping evaluation.")
        _emit(_job(state), "evaluator", "completed", "Evaluation skipped (module missing)")
        return {"quality_evaluation": {"error": "Module missing"}}
    except Exception as e:
        logger.warning(f"Evaluation Error: {e}")
        _emit(_job(state), "evaluator", "error", f"Evaluation failed: {str(e)}")
        return {"quality_evaluation": {"error": str(e)}}
