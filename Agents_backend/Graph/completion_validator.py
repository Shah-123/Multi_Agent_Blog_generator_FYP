"""
Completion Validator
Ensures all sections are complete and properly formatted.
"""

import re
from typing import Dict, List

def validate_completion(state: dict) -> dict:
    """
    Validates that the blog is complete and properly formatted.
    """
    print("--- ✅ VALIDATING COMPLETENESS ---")
    
    final_text = state.get("final", "")
    plan = state.get("plan")
    
    if not plan:
        print("   ⚠️ No plan found")
        return {}
    
    issues = []
    
    # 1. Check if all sections exist
    expected_sections = len(plan.tasks)
    actual_sections = len(re.findall(r'^## ', final_text, re.MULTILINE))
    
    if actual_sections < expected_sections:
        issues.append(f"Missing sections: Expected {expected_sections}, found {actual_sections}")
        print(f"   ⚠️ Missing {expected_sections - actual_sections} sections")
    
    # 2. Check for incomplete sentences
    paragraphs = final_text.split('\n\n')
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if para and len(para) > 50:  # Ignore short lines
            if not para.endswith(('.', '!', '?', '"', ')')):
                issues.append(f"Paragraph {i+1} may be incomplete")
                print(f"   ⚠️ Incomplete paragraph detected")
    
    # 3. Check word count
    total_words = len(final_text.split())
    expected_words = sum(task.target_words for task in plan.tasks)
    
    if total_words < (expected_words * 0.8):  # Less than 80% of expected
        issues.append(f"Word count low: {total_words} words (expected ~{expected_words})")
        print(f"   ⚠️ Word count: {total_words}/{expected_words} (low)")
    else:
        print(f"   ✅ Word count: {total_words} words")
    
    # 4. Check for broken image placeholders
    broken_images = re.findall(r'\[\[IMAGE_\d+\]\]', final_text)
    if broken_images:
        issues.append(f"Found {len(broken_images)} unprocessed image placeholders")
        print(f"   ⚠️ {len(broken_images)} broken image placeholders")
    
    # Generate completion report
    if issues:
        report = "COMPLETION VALIDATION REPORT\n"
        report += "=" * 60 + "\n"
        report += "⚠️ ISSUES FOUND:\n\n"
        for i, issue in enumerate(issues, 1):
            report += f"{i}. {issue}\n"
    else:
        report = "COMPLETION VALIDATION REPORT\n"
        report += "=" * 60 + "\n"
        report += "✅ Blog is complete and properly formatted!\n"
        report += f"- Total words: {total_words}\n"
        report += f"- Sections: {actual_sections}/{expected_sections}\n"
    
    return {"completion_report": report}