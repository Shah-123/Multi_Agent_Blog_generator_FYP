from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State
from .utils import logger, llm, _job, _emit

# ✅ FIX #10: Summarize the full blog into a structured brief BEFORE passing
# to campaign agents. Previously blog_post[:4000] was used, which for a
# 3000-word blog only covered ~600 words — roughly just the introduction.
# Campaign agents were writing LinkedIn posts, emails, and landing pages
# based entirely on the intro, missing all core arguments and the CTA.

CAMPAIGN_BRIEF_SYSTEM = """You are a content strategist. 
Read the full blog post and extract a structured campaign brief.

Return EXACTLY this format — no extra text:

TITLE: [exact blog title]
CORE_PROBLEM: [1 sentence — what problem does this solve?]
KEY_ARGUMENTS:
- [argument 1]
- [argument 2]
- [argument 3]
KEY_STATS:
- [most compelling stat or fact]
- [second most compelling stat or fact]
- [third most compelling stat or fact]
PRACTICAL_TIPS:
- [actionable tip 1]
- [actionable tip 2]
- [actionable tip 3]
PRIMARY_CTA: [1 sentence — what should the reader do after reading?]
TARGET_AUDIENCE: [who is this written for?]
TONE: [what is the writing tone?]
"""


def _build_campaign_brief(blog_post: str, topic: str, evidence: list) -> str:
    """
    Summarizes the full blog into a structured brief for campaign agents.
    Uses the full blog text — no character truncation.
    Falls back to a truncated version only if the blog exceeds 100k chars
    (which would be an unusually large document).
    """
    # Safety cap at 100k chars (~15k words) — well beyond any normal blog
    safe_blog = blog_post[:100_000]

    key_stats = "\n".join(
        [f"- {e.snippet[:120]}... ({e.url})" for e in evidence[:5]]
    )

    response = llm.invoke([
        SystemMessage(content=CAMPAIGN_BRIEF_SYSTEM),
        HumanMessage(content=(
            f"TOPIC: {topic}\n\n"
            f"FULL BLOG POST:\n{safe_blog}\n\n"
            f"SUPPORTING EVIDENCE:\n{key_stats}"
        ))
    ])

    return response.content.strip()


def campaign_generator_node(state: State) -> dict:
    _emit(_job(state), "campaign_generator", "started", "Generating 6-part omnichannel campaign...")
    logger.info("🚀 GENERATING OMNICHANNEL CAMPAIGN PACK ---")

    from Graph.templates import (
        LINKEDIN_SYSTEM, YOUTUBE_SYSTEM, FACEBOOK_SYSTEM,
        EMAIL_SEQUENCE_SYSTEM, TWITTER_THREAD_SYSTEM, LANDING_PAGE_SYSTEM
    )

    blog_post = state["final"]
    topic = state["topic"]
    evidence = state.get("evidence", [])

    # ✅ FIX #10: Build a full structured brief from the entire blog post.
    # All 6 campaign agents receive this brief instead of a raw [:4000] slice.
    logger.info("📋 Building campaign brief from full blog content...")
    _emit(_job(state), "campaign_generator", "working", "Summarizing full blog for campaign brief...")

    campaign_brief = _build_campaign_brief(blog_post, topic, evidence)
    logger.info("✅ Campaign brief ready.")

    context = (
        f"TOPIC: {topic}\n\n"
        f"CAMPAIGN BRIEF (summarized from full blog):\n{campaign_brief}"
    )

    def _gen(system_prompt):
        return llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=context)
        ]).content

    with ThreadPoolExecutor(max_workers=6) as pool:
        linkedin_future = pool.submit(_gen, LINKEDIN_SYSTEM)
        youtube_future  = pool.submit(_gen, YOUTUBE_SYSTEM)
        facebook_future = pool.submit(_gen, FACEBOOK_SYSTEM)
        email_future    = pool.submit(_gen, EMAIL_SEQUENCE_SYSTEM)
        twitter_future  = pool.submit(_gen, TWITTER_THREAD_SYSTEM)
        landing_future  = pool.submit(_gen, LANDING_PAGE_SYSTEM)

    linkedin = linkedin_future.result()
    youtube  = youtube_future.result()
    facebook = facebook_future.result()
    email    = email_future.result()
    twitter  = twitter_future.result()
    landing  = landing_future.result()

    logger.info("✅ All 6 Campaign Assets Generated")
    _emit(_job(state), "campaign_generator", "completed", "Generated Email, Twitter, Landing Page, LinkedIn, YouTube & Facebook content", {"assets": 6})

    return {
        "linkedin_post":  linkedin,
        "youtube_script": youtube,
        "facebook_post":  facebook,
        "email_sequence": email,
        "twitter_thread": twitter,
        "landing_page":   landing,
    }