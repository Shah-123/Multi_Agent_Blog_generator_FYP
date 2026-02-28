import os
from typing import List
import requests
from bs4 import BeautifulSoup
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from Graph.state import State, EvidencePack
from Graph.templates import RESEARCH_SYSTEM
from .utils import logger, llm, _job, _emit

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """Safe Tavily search wrapper."""
    if not os.getenv("TAVILY_API_KEY"):
        logger.warning("TAVILY_API_KEY missing. Skipping search.")
        return []
    try:
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content") or r.get("snippet", ""),
                "published_at": r.get("published_date") or r.get("published_at"),
                "source": r.get("source"),
            })
        return out
    except Exception as e:
        logger.warning(f"Search failed for '{query}': {e}")
        return []

def scrape_full_webpage_fallback(url: str, max_words: int = 1500) -> str:
    """Fallback: visits a URL and scrapes the actual article text using BeautifulSoup."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style", "nav", "footer", "aside"]):
            script.decompose()

        text = soup.get_text(separator=' ', strip=True)
        words = text.split()
        return " ".join(words[:max_words])
    
    except Exception as e:
        logger.warning(f"Fallback scrape failed for {url}: {e}")
        return ""

def scrape_full_webpage(url: str, max_words: int = 1500) -> str:
    """Primary: Converts URL to clean Markdown using Jina Reader API for optimal LLM ingestion."""
    try:
        # Prevent Jina from trying to read itself if a URL is broken
        if not url.startswith("http"):
            return ""
            
        jina_url = f"https://r.jina.ai/{url}"
        headers = {'Accept': 'text/event-stream'} # Jina prefers this
        
        logger.info(f"Using Jina Reader for: {url}")
        response = requests.get(jina_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            text = response.text
            words = text.split()
            # If Jina returns too little, maybe it hit a block. Try fallback.
            if len(words) < 50:
                logger.warning(f"Jina returned very little content for {url}. Attempting fallback.")
                return scrape_full_webpage_fallback(url, max_words)
            return " ".join(words[:max_words])
            
        else:
            logger.warning(f"Jina Reader failed ({response.status_code}) for {url}. Attempting fallback.")
            return scrape_full_webpage_fallback(url, max_words)
            
    except Exception as e:
        logger.warning(f"Jina Reader exception on {url}: {e}. Attempting fallback.")
        return scrape_full_webpage_fallback(url, max_words)

def research_node(state: State) -> dict:
    _emit(_job(state), "research", "started", "Searching the web for evidence...")
    logger.info("🔍 DEEP RESEARCHING ---")
    queries = (state.get("queries") or [])[:5] 
    
    found_urls = set()
    raw_results = []
    
    for q in queries:
        logger.info(f"Searching: {q}")
        _emit(_job(state), "research", "working", f"Searching: {q}")
        results = _tavily_search(q, max_results=3)
        for r in results:
            if r['url'] not in found_urls:
                found_urls.add(r['url'])
                raw_results.append(r)

    if not raw_results:
        logger.warning("No results found.")
        _emit(_job(state), "research", "completed", "No results found", {"sources": 0})
        return {"evidence": []}

    logger.info(f"🕸️ Scraping {len(raw_results[:10])} top articles...")
    _emit(_job(state), "research", "working", f"Deep-scraping {len(raw_results[:10])} articles...")
    
    deep_evidence_context = ""
    for idx, r in enumerate(raw_results[:10]):
        logger.info(f"-> Reading: {r['url']}")
        full_text = scrape_full_webpage(r['url'])
        
        if full_text:
            deep_evidence_context += f"SOURCE {idx+1}: {r['title']} ({r['url']})\n"
            deep_evidence_context += f"CONTENT: {full_text[:3000]}\n\n" 

    logger.info("🧠 Analyzing full articles for hard evidence...")
    _emit(_job(state), "research", "working", "Extracting verified facts from articles...")
    
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Read the following full articles and extract ONLY hard facts, statistics, and verifiable claims.\n\n"
            f"SCRAPED ARTICLES:\n{deep_evidence_context}" 
        )),
    ])

    evidence = list({e.url: e for e in pack.evidence if e.url}.values())
    
    logger.info(f"✅ Extracted {len(evidence)} verified deep-evidence items.")
    _emit(_job(state), "research", "completed", f"Found {len(evidence)} verified evidence items", {"sources": len(evidence)})
    return {"evidence": evidence}
