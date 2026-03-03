import os
import sys
import time
import requests
import tempfile
import wave
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from Graph.state import State
from .utils import logger, llm, _job, _emit

class VideoScenePlan(BaseModel):
    """Schema for planning stock video searches."""
    keywords: List[str] = Field(description="3-5 highly specific, visually descriptive search queries (e.g., 'doctor looking at tablet', 'futuristic data server')")

VIDEO_PLAN_SYSTEM = """You are a video producer. Your goal is to plan the visual B-roll footage for a "Faceless" short-form video (TikTok/Reels).
Based on the topic and the voiceover script provided, generate 3-5 specific visual search queries.
These queries will be used to search a stock video site like Pexels for background footage.
Focus on moody, aesthetic, or highly relevant visual concepts. Keep queries to 2-4 words maximum.
"""

VOICEOVER_SYSTEM_PROMPT = """You are a professional YouTube Shorts / TikTok scriptwriter.
Given the blog post summary, write a highly engaging, fast-paced voiceover script to be read by a single narrator.
The script should be around 45 to 60 seconds long when spoken (about 120-150 words).
Ensure:
- Strong hook at the beginning
- Clear, punchy educational points covering the FULL blog (not just the intro)
- A call to action at the end
- DO NOT INCLUDE ANY SPEAKER LABELS LIKE "NARRATOR:" OR "VOICEOVER:". Just output the raw text to be spoken.
"""

VOICEOVER_BRIEF_SYSTEM = """You are a content strategist preparing a brief for a short-form video scriptwriter.
Read the full blog post and extract a structured video brief.

Return EXACTLY this format — no extra text:

TITLE: [exact blog title]
CORE_HOOK: [1 sentence — the most attention-grabbing fact or claim in this blog]
KEY_POINTS:
- [most important point from the blog]
- [second most important point]
- [third most important point]
SURPRISING_STAT: [the single most compelling statistic or finding, if any]
PRIMARY_CTA: [what should the viewer do after watching?]
TONE: [what is the writing tone?]
"""

# ✅ Retry configuration for Gemini TTS.
# Gemini's TTS endpoint occasionally returns transient 500 errors.
# 3 attempts with exponential backoff (2s, 4s) is enough to survive
# most transient failures without blocking the pipeline for too long.
_TTS_MAX_ATTEMPTS = 3
_TTS_BACKOFF_BASE = 2  # seconds — attempt 1: 2s wait, attempt 2: 4s wait


def _build_voiceover_brief(blog_content: str, topic: str) -> str:
    """
    Summarizes the full blog into a structured brief for the voiceover scriptwriter.
    Mirrors _build_campaign_brief() in campaign.py.
    Safety cap at 100k chars — well beyond any normal blog.
    """
    safe_blog = blog_content[:100_000]

    response = llm.invoke([
        SystemMessage(content=VOICEOVER_BRIEF_SYSTEM),
        HumanMessage(content=(
            f"TOPIC: {topic}\n\n"
            f"FULL BLOG POST:\n{safe_blog}"
        ))
    ])

    return response.content.strip()


def fetch_pexels_video(query: str, download_dir: str, index: int) -> Optional[str]:
    """Fetches a vertical video from Pexels and saves it to disk."""
    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        logger.warning("No Pexels API key found.")
        return None

    headers = {"Authorization": api_key}
    url = f"https://api.pexels.com/videos/search?query={query}&per_page=3&orientation=portrait"

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data.get('videos'):
            logger.warning(f"No videos found on Pexels for query: {query}")
            return None

        video = data['videos'][0]
        best_file = None
        for file in video.get('video_files', []):
            if file.get('quality') == 'hd' and file.get('height', 0) >= 1080:
                best_file = file
                break

        if not best_file and video.get('video_files'):
            best_file = video['video_files'][0]

        if not best_file:
            return None

        link = best_file['link']
        vid_path = os.path.join(download_dir, f"clip_{index}.mp4")
        with requests.get(link, stream=True) as r:
            r.raise_for_status()
            with open(vid_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return vid_path

    except Exception as e:
        logger.error(f"Error fetching Pexels video for '{query}': {e}")
        return None


# ======================================================================
# VOICEOVER HELPERS
# ======================================================================

def save_pcm_as_wav(pcm_bytes: bytes, output_path: str):
    """Saves raw PCM bytes from Gemini to a WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)


def generate_tts_voiceover(text: str, voice: str = "Puck") -> Optional[str]:
    """
    Generate TTS audio for the voiceover using Gemini and save to WAV.

    ✅ FIX: Added retry with exponential backoff.
    Gemini's TTS endpoint returns transient 500 INTERNAL errors that succeed
    on a subsequent attempt. Previously the function caught the exception,
    logged it, and returned None immediately — aborting the entire video
    pipeline on what is almost always a recoverable API hiccup.

    Retry schedule (3 attempts total):
      Attempt 1 → fails → wait 2s
      Attempt 2 → fails → wait 4s
      Attempt 3 → fails → return None (genuine failure)
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY is not set. Cannot generate voiceover.")
        return None

    gemini_client = genai.Client(api_key=api_key)

    for attempt in range(1, _TTS_MAX_ATTEMPTS + 1):
        try:
            logger.info(f"   🔊 TTS attempt {attempt}/{_TTS_MAX_ATTEMPTS} (voice: {voice})...")

            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash-preview-tts',
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    )
                )
            )

            audio_bytes = None
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    audio_bytes = part.inline_data.data
                    break

            if audio_bytes:
                temp_wav = tempfile.mktemp(suffix=".wav")
                save_pcm_as_wav(audio_bytes, temp_wav)
                logger.info(f"   ✅ TTS succeeded on attempt {attempt}.")
                return temp_wav

            # Response came back but contained no audio data — not worth retrying
            logger.error("Gemini returned a response but contained no audio data.")
            return None

        except Exception as e:
            if attempt < _TTS_MAX_ATTEMPTS:
                wait = _TTS_BACKOFF_BASE * attempt  # 2s, 4s
                logger.warning(
                    f"   ⚠️ TTS attempt {attempt} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.error(
                    f"   ❌ TTS failed after {_TTS_MAX_ATTEMPTS} attempts. "
                    f"Last error: {e}"
                )

    return None



# ======================================================================
# MAIN VIDEO GENERATOR NODE
# ======================================================================

def video_generator_node(state: State) -> dict:
    """Generates a stock video short by writing a voiceover script, generating TTS,
    fetching Pexels clips, and merging them with audio."""
    from langchain_openai import ChatOpenAI

    _emit(_job(state), "video", "started", "Structuring voiceover script for Short...")
    logger.info("🎬 GENERATING STOCK VIDEO ---")

    topic        = state.get("topic", "Unknown")
    blog_content = state.get("final", "")

    if not blog_content:
        logger.warning("No blog content found. Skipping video generation.")
        _emit(_job(state), "video", "error", "Skipped: Requires generated blog content.")
        return {"video_path": None}

    video_dir = Path("generated_videos")
    video_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir  = tempfile.mkdtemp()

    # 1. Build structured brief from full blog
    _emit(_job(state), "video", "working", "Summarizing full blog for voiceover brief...")
    logger.info("📋 Building voiceover brief from full blog content...")
    voiceover_brief = _build_voiceover_brief(blog_content, topic)
    logger.info("✅ Voiceover brief ready.")

    # 2. Write voiceover script from brief
    _emit(_job(state), "video", "working", "Generating voiceover script...")
    text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    response = text_llm.invoke([
        SystemMessage(content=VOICEOVER_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"TOPIC: {topic}\n\n"
            f"BLOG SUMMARY BRIEF:\n{voiceover_brief}"
        ))
    ])
    voiceover_script = response.content.strip()

    # 3. Generate audio (with retry — see generate_tts_voiceover)
    _emit(_job(state), "video", "working", "Synthesizing voiceover audio with Gemini...")
    audio_path = generate_tts_voiceover(voiceover_script, voice="Puck")

    if not audio_path:
        logger.error("Audio generation failed after all retries.")
        _emit(_job(state), "video", "error", "Audio generation failed after retries.")
        return {"video_path": None}

    try:
        from moviepy import AudioFileClip
        audio_clip     = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return {"video_path": None}

    # 5. Plan video scenes
    _emit(_job(state), "video", "working", "Planning stock video background...")
    planner = llm.with_structured_output(VideoScenePlan)
    plan    = planner.invoke([
        SystemMessage(content=VIDEO_PLAN_SYSTEM),
        HumanMessage(content=f"Topic: {topic}\n\nVoiceover Script:\n{voiceover_script}")
    ])

    queries = plan.keywords if plan.keywords else [topic, "abstract background", "technology"]
    logger.info(f"   🎥 Video Queries: {queries}")
    _emit(_job(state), "video", "working", f"Fetching {len(queries)} stock videos from Pexels...")

    # 6. Fetch from Pexels
    downloaded_clips = []
    for i, q in enumerate(queries):
        clip_path = fetch_pexels_video(q, temp_dir, i)
        if clip_path:
            downloaded_clips.append(clip_path)
        if len(downloaded_clips) * 15 > audio_duration + 10:
            logger.info("   ✅ Gathered enough footage for the duration.")
            break

    if not downloaded_clips:
        logger.warning("No specific clips found, trying generic fallback...")
        fallback = fetch_pexels_video("abstract minimalist", temp_dir, 99)
        if fallback:
            downloaded_clips.append(fallback)

    if not downloaded_clips:
        logger.error("Failed to download any stock videos. Skipping.")
        _emit(_job(state), "video", "error", "Failed to fetch stock footage.")
        return {"video_path": None}

    # 7. Stitch with moviepy
    _emit(_job(state), "video", "working", "Stitching footage and overlaying audio...")
    logger.info("   ✂️ Editing video with moviepy...")

    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
        import moviepy.video.fx as vfx
    except ImportError:
        logger.error("moviepy not installed. Skipping video generation.")
        _emit(_job(state), "video", "error", "Missing moviepy dependency.")
        return {"video_path": None}

    try:
        video_clips = []
        for p in downloaded_clips:
            try:
                clip = VideoFileClip(p)
                chunk_length = min(10, clip.duration)
                clip = clip.subclipped(0, chunk_length).resized(width=1080)
                video_clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to load clip {p}: {e}")

        if not video_clips:
            raise Exception("No valid video clips could be loaded by moviepy.")

        try:
            final_video = concatenate_videoclips(video_clips, method="compose")
        except Exception:
            final_video = concatenate_videoclips(video_clips)

        if final_video.duration < audio_duration:
            final_video = final_video.with_effects([vfx.Loop(duration=audio_duration)])
        else:
            final_video = final_video.subclipped(0, audio_duration)



        final_video = final_video.with_audio(audio_clip)

        output_file = str(video_dir / f"short_{timestamp}.mp4")
        logger.info(f"   🎬 Exporting to {output_file}...")

        final_video.write_videofile(
            output_file,
            fps=30,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            logger=None
        )

        for clip in video_clips:
            clip.close()
        audio_clip.close()
        final_video.close()

        try:
            os.remove(audio_path)
        except Exception:
            pass

        logger.info("   ✅ Video Generation Complete!")
        _emit(_job(state), "video", "completed", "Stock video generated successfully.")
        return {"video_path": output_file}

    except Exception as e:
        logger.error(f"Video editing failed: {e}")
        _emit(_job(state), "video", "error", f"Video edit failed: {e}")
        return {"video_path": None}