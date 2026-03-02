import os
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types

from Graph.agents.utils import logger, _job, _emit

# ✅ FIX #2: Removed module-level client initialization.
# Previously: api_key = os.getenv("GEMINI_API_KEY") ran at import time,
# before load_dotenv() in main.py, so gemini_client was always None.
# Also renamed from GEMINI_API_KEY → GOOGLE_API_KEY to match
# multimedia.py and video.py — all three files now use one consistent key name.

# ============================================================================
# PODCAST AUDIO GENERATION
# ============================================================================

PODCAST_SYSTEM_PROMPT = """You are an engaging solo podcast host discussing a blog post topic.
Create a 2-3 minute audio clip naturally discussing the core ideas.
Make it conversational, educational, and natural-sounding. Include pauses like "hmm", "you know", "right".
Speak directly to the audience.

RULES:
- Talk for about 2-3 minutes.
- Tone: Educational but entertaining.
"""

def _get_gemini_client():
    """
    Lazy initialization of the Gemini client.
    Called at function runtime (not import time), ensuring
    load_dotenv() in main.py has already run before we read the env var.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def generate_podcast_audio(state: dict, output_path: str) -> bool:
    """Generate a conversational podcast audio from blog content using Gemini's native audio output."""
    
    # ✅ FIX #2: Client is now created here at call time, not at module import time.
    gemini_client = _get_gemini_client()
    
    if not gemini_client:
        logger.warning("GOOGLE_API_KEY missing. Cannot generate podcast.")
        return False
        
    plan = state.get("plan")
    topic = state.get("topic", "the topic")
    
    # Build context from sections
    sections_summary = ""
    if plan and hasattr(plan, 'tasks'):
        sections_summary = "\n".join([f"- {task.title}" for task in plan.tasks])
    
    prompt = f"""Create an audio podcast discussing: "{plan.blog_title if plan else topic}"

Key sections to cover:
{sections_summary}

Target audience: {plan.audience if plan else "general"}
Tone: {plan.tone if plan else "conversational"}
"""

    try:
        logger.info("🎙️ Requesting Gemini 2.5 Flash for native audio generation...")
        
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash-preview-tts',
            contents=[
               prompt
            ],
            config=types.GenerateContentConfig(
                system_instruction=PODCAST_SYSTEM_PROMPT,
                response_modalities=["AUDIO"],
                temperature=0.7
            )
        )
        
        audio_data = None
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    audio_data = part.inline_data.data
                    break
                    
        if audio_data:
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return True
        else:
            logger.error("No audio data returned in Gemini response.")
            return False
            
    except Exception as e:
        logger.error(f"Gemini Audio generation failed: {e}")
        return False

# ============================================================================
# MAIN NODE
# ============================================================================

def podcast_node(state: dict) -> dict:
    """
    Generate podcast audio from blog content using Gemini 2.5 Flash.
    Returns: {"podcast_audio_path": str}
    """
    _emit(_job(state), "podcast_generator", "started", "Generating AI Podcast Audio via Gemini...")
    logger.info("--- 🎙️ PODCAST STATION (GEMINI) ---")
    
    # Setup
    podcast_dir = Path("generated_podcasts")
    podcast_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = podcast_dir / f"podcast_{timestamp}.wav"
    
    # Generate Audio
    logger.info(f"   ✍️ Synthesizing Podcast Audio...")
    success = generate_podcast_audio(state, str(final_path))
    
    if success:
        logger.info(f"   ✅ Podcast saved to {final_path}")
        _emit(_job(state), "podcast_generator", "completed", "Podcast audio generated successfully.")
        return {"podcast_audio_path": str(final_path)}
    else:
        logger.error(f"   ❌ Podcast generation failed.")
        _emit(_job(state), "podcast_generator", "error", "Failed to generate podcast audio.")
        return {"podcast_audio_path": None}