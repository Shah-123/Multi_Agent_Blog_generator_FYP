"""
Keyword Optimization Module
Analyzes and optimizes keyword usage in blog content.
"""

import re
from typing import Dict, List

def analyze_keyword_density(text: str, keywords: List[str]) -> Dict:
    """
    Analyze how well keywords are integrated in the content.
    
    Returns a dict with metrics for each keyword:
    - count: number of occurrences
    - density: percentage (industry standard: 1-2%)
    - in_title: present in first 200 chars
    - in_first_paragraph: present in first 500 chars
    - in_headings: present in any heading
    - status: optimal/low/high
    """
    
    text_lower = text.lower()
    word_count = len(text.split())
    
    results = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        # Count occurrences
        count = text_lower.count(keyword_lower)
        
        # Calculate density (industry standard: 1-2%)
        density = (count / word_count * 100) if word_count > 0 else 0
        
        # Check strategic placement
        in_title = keyword_lower in text[:200].lower()
        in_first_para = keyword_lower in text[:500].lower()
        
        # Check if in headings (lines starting with #)
        in_headings = bool(re.search(
            rf'^##+ .*{re.escape(keyword_lower)}.*$', 
            text, 
            re.MULTILINE | re.IGNORECASE
        ))
        
        # Determine status
        if 1 <= density <= 2.5:
            status = "optimal"
        elif density < 1:
            status = "low"
        else:
            status = "high"
        
        results[keyword] = {
            "count": count,
            "density": round(density, 2),
            "in_title": in_title,
            "in_first_paragraph": in_first_para,
            "in_headings": in_headings,
            "status": status
        }
    
    return results


def generate_optimization_suggestions(analysis: Dict) -> List[str]:
    """Generate actionable suggestions based on keyword analysis."""
    
    suggestions = []
    
    for keyword, stats in analysis.items():
        # Density suggestions
        if stats["status"] == "low":
            suggestions.append(
                f"• '{keyword}' appears only {stats['count']} times ({stats['density']}%). "
                f"Consider adding 1-2 more natural mentions to reach 1-2% density."
            )
        elif stats["status"] == "high":
            suggestions.append(
                f"• '{keyword}' appears {stats['count']} times ({stats['density']}%). "
                f"This may trigger keyword stuffing penalties. Consider removing some instances."
            )
        
        # Placement suggestions
        if not stats["in_title"]:
            suggestions.append(
                f"• '{keyword}' is missing from the title/introduction. "
                f"Add it to the first paragraph for better SEO."
            )
        
        if not stats["in_headings"]:
            suggestions.append(
                f"• '{keyword}' is not used in any headings (H2/H3). "
                f"Consider incorporating it into at least one subheading."
            )
    
    return suggestions


def keyword_optimizer_node(state: dict) -> dict:
    """
    Analyzes and optimizes keyword usage in the final blog.
    
    This node:
    1. Analyzes keyword density and placement
    2. Generates optimization suggestions
    3. Creates a comprehensive keyword report
    """
    
    print("--- 🎯 OPTIMIZING KEYWORDS ---")
    
    final_text = state.get("final", "")
    target_keywords = state.get("target_keywords", [])
    
    # Skip if no keywords specified
    if not target_keywords:
        print("   ⏭️ No keywords specified, skipping optimization.")
        return {}
    
    if not final_text:
        print("   ⚠️ No content to analyze yet.")
        return {}
    
    # Analyze current keyword usage
    analysis = analyze_keyword_density(final_text, target_keywords)
    
    # Generate optimization suggestions
    suggestions = generate_optimization_suggestions(analysis)
    
    # Create comprehensive report
    report = "KEYWORD OPTIMIZATION REPORT\n"
    report += "=" * 60 + "\n\n"
    
    # Overall summary
    optimal_count = sum(1 for stats in analysis.values() if stats["status"] == "optimal")
    report += f"📊 SUMMARY: {optimal_count}/{len(target_keywords)} keywords are optimally integrated\n\n"
    
    # Detailed breakdown for each keyword
    for keyword, stats in analysis.items():
        status_emoji = "✅" if stats["status"] == "optimal" else "⚠️" if stats["status"] == "low" else "❌"
        
        report += f"{status_emoji} {keyword.upper()}\n"
        report += f"   Occurrences: {stats['count']}\n"
        report += f"   Density: {stats['density']}% (optimal: 1-2%)\n"
        report += f"   In Title/Intro: {'✓' if stats['in_title'] else '✗'}\n"
        report += f"   In First Paragraph: {'✓' if stats['in_first_paragraph'] else '✗'}\n"
        report += f"   In Headings: {'✓' if stats['in_headings'] else '✗'}\n"
        report += f"   Status: {stats['status'].upper()}\n\n"
    
    # Add suggestions section
    if suggestions:
        report += "💡 OPTIMIZATION SUGGESTIONS\n"
        report += "-" * 60 + "\n"
        report += "\n".join(suggestions)
        report += "\n"
    else:
        report += "✅ EXCELLENT! All keywords are optimally integrated.\n"
        report += "No further optimization needed.\n"
    
    # Print summary to console
    print(f"   📊 Analyzed {len(target_keywords)} keywords")
    print(f"   ✅ {optimal_count} optimal, ⚠️ {len(target_keywords) - optimal_count} need attention")
    
    return {
        "keyword_analysis": analysis,
        "keyword_report": report
    }


def get_keyword_summary(analysis: Dict) -> str:
    """Generate a one-line summary of keyword optimization status."""
    
    if not analysis:
        return "No keywords analyzed"
    
    optimal = sum(1 for stats in analysis.values() if stats["status"] == "optimal")
    total = len(analysis)
    
    if optimal == total:
        return f"✅ All {total} keywords optimally integrated"
    elif optimal == 0:
        return f"⚠️ 0/{total} keywords optimized - needs work"
    else:
        return f"⚠️ {optimal}/{total} keywords optimized"