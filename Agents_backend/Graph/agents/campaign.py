from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State
from .utils import logger, llm, _job, _emit

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
    
    key_stats = "\n".join([f"- {e.snippet[:100]}... ({e.url})" for e in evidence[:5]])
    
    context = f"TOPIC: {topic}\nBLOG CONTENT:\n{blog_post[:4000]}\nSTATS:\n{key_stats}"
    
    def _gen(system_prompt):
        return llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=context)]).content
    
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
        "linkedin_post": linkedin,
        "youtube_script": youtube,
        "facebook_post": facebook,
        "email_sequence": email,
        "twitter_thread": twitter,
        "landing_page": landing,
    }
