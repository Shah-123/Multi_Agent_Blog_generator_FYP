from langchain_core.messages import SystemMessage, HumanMessage
import re

from Graph.state import State
from Graph.structured_data import QAReport
from .utils import logger, llm_quality, _job, _emit

QA_AGENT_SYSTEM = """You are an elite Quality Assurance (QA) Editor for a top-tier publishing platform.
Your job is to read the provided blog post and conduct a rigorous final audit before publication.

You must evaluate the content on three dimensions:
1. FACTS & ACCURACY: Are there hallucinations? Did the writer invent claims not supported by the evidence?
2. COMPLETENESS & STRUCTURE: Does it flow logically? Did the writer adequately address the initial plan's goals? Are there missing sections?
3. READABILITY & TONE: Is the writing engaging, professional, and free of robotic cliches (e.g., "In conclusion", "It is important to note")?

INSTRUCTIONS:
- Be highly critical. Do not give out 9/10 or 10/10 scores easily.
- If you find CRITICAL issues (false facts, major hallucinated claims, or significantly incomplete sections), set the verdict to NEEDS_REVISION.
- Return a structured QA Report answering these exact dimensions.
"""

def qa_agent_node(state: State) -> dict:
    _emit(_job(state), "qa_agent", "started", "Running comprehensive QA audit (Format + Facts + Quality)...")
    logger.info("🕵️ QUALITY ASSURANCE AUDIT ---")
    
    # --- 1. PRE-LLM LEXICAL CHECKS & AUTO-FIXES ---
    plan = state.get("plan")
    final_text = state.get("final", "")
    
    # Combine sections if final somehow doesn't exist yet
    if not final_text:
        raw_sections = state.get("sections", [])
        sections = list(raw_sections)
        sections.sort(key=lambda x: x[0] if isinstance(x, (tuple, list)) and len(x) > 0 else 0)
        final_text = "\n\n".join([str(s[1]) for s in sections if isinstance(s, (tuple, list)) and len(s) > 1])
        
    fixes_applied = []
    lexical_issues = []
    
    if plan:
        expected_sections = len(plan.tasks)
        actual_sections = len(re.findall(r'^#{2,4} ', final_text, re.MULTILINE))
        if actual_sections < expected_sections:
            lexical_issues.append(f"Missing sections (Expected {expected_sections}, found {actual_sections})")
            
    # Auto-fix incomplete sentences at end of paragraphs
    paragraphs = final_text.split('\n\n')
    fixed_paragraphs = []
    for i, para in enumerate(paragraphs):
        original = para
        para_stripped = para.strip()
        if para_stripped.startswith('![') or para_stripped.startswith('#'):
            fixed_paragraphs.append(original)
            continue
        clean_para = para_stripped.strip('*_` ')
        if clean_para and len(clean_para) > 50:
            if not clean_para.endswith(('.', '!', '?', '"', ')')):
                fixed_paragraphs.append(para.rstrip() + ".")
                fixes_applied.append(f"Added missing period to paragraph {i+1}")
                continue
        fixed_paragraphs.append(original)
        
    final_text = '\n\n'.join(fixed_paragraphs)
    
    # Auto-fix broken image placeholders
    broken_images = re.findall(r'\[\[IMAGE_\d+\]\]', final_text)
    if broken_images:
        final_text = re.sub(r'\[\[IMAGE_\d+\]\]\n?', '', final_text)
        fixes_applied.append(f"Removed {len(broken_images)} broken image placeholder(s)")
        
    # --- 2. LLM QA AUDIT ---
    checker = llm_quality.with_structured_output(QAReport)
    
    evidence_summary = "\n".join([
        f"- {e.title} ({e.url})\n  Content: {e.snippet[:500]}..."
        for e in state.get("evidence", [])[:15]
    ])
    
    report = checker.invoke([
        SystemMessage(content=QA_AGENT_SYSTEM),
        HumanMessage(content=(
            f"BLOG CONTENT TO AUDIT:\n{final_text[:12000]}\n\n"
            f"EVIDENCE USED IN RESEARCH:\n{evidence_summary}"
        ))
    ])
    
    # --- 3. FORMAT REPORT ---
    report_text = f"QA AUDIT REPORT\n"
    report_text += "=" * 60 + "\n"
    report_text += f"Overall Score: {report.overall_score}/10\n"
    report_text += f"Verdict: {report.verdict}\n\n"
    
    report_text += "Metrics:\n"
    report_text += f"- Depth: {report.depth_score}/10\n"
    report_text += f"- Structure: {report.structure_score}/10\n"
    report_text += f"- Readability: {report.readability_score}/10\n\n"
    
    if report.strengths:
        report_text += "Strengths:\n"
        for s in report.strengths:
            report_text += f"• {s}\n"
        report_text += "\n"
    
    if fixes_applied:
        report_text += "🔧 Auto-Fixes Applied During Formatting:\n"
        for fix in fixes_applied:
            report_text += f"• {fix}\n"
        report_text += "\n"
        
    if lexical_issues:
        report_text += "⚠️ Formatting Issues Detected:\n"
        for iss in lexical_issues:
            report_text += f"• {iss}\n"
        report_text += "\n"
    
    if report.issues:
        report_text += f"Content Issues Found: {len(report.issues)}\n"
        for i, issue in enumerate(report.issues, 1):
            report_text += f"{i}. [{issue.issue_type}] {issue.claim}\n"
            report_text += f"   -> Fix: {issue.recommendation}\n\n"
    else:
        report_text += "✅ No content issues found!\n"
    
    logger.info(f"📊 Overall Score: {report.overall_score}/10 | Verdict: {report.verdict}")
    _emit(_job(state), "qa_agent", "completed", f"QA Score: {report.overall_score}/10 — {report.verdict}", {"score": report.overall_score, "verdict": report.verdict, "issues": len(report.issues) if report.issues else 0})
    
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
        "final": final_text,  # Save the auto-fixed version
        "qa_report": report_text,
        "qa_verdict": report.verdict,
        "qa_issues": issues_list,
        "qa_score": report.overall_score,
    }

