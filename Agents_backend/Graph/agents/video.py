import os
import sys
import requests
import tempfile
import wave
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

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
Given the blog post, write a highly engaging, fast-paced voiceover script to be read by a single narrator.
The script should be around 45 to 60 seconds long when spoken (about 120-150 words).
Ensure:
- Strong hook at the beginning
- Clear, punchy educational points
- A call to action at the end
- DO NOT INCLUDE ANY SPEAKER LABELS LIKE "NARRATOR:" OR "VOICEOVER:". Just output the raw text to be spoken.
"""

def fetch_pexels_video(query: str, download_dir: str, index: int) -> str:
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
            
        # Get the first video, and find a good quality file (HD)
        video = data['videos'][0]
        best_file = None
        for file in video.get('video_files', []):
            if file.get('quality') == 'hd' and file.get('height', 0) >= 1080:
                best_file = file
                break
        
        # Fallback to first file if no HD found
        if not best_file and video.get('video_files'):
            best_file = video['video_files'][0]
            
        if not best_file:
            return None
            
        link = best_file['link']
        
        # Download the file
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
# VOICEOVER & CAPTION HELPERS
# ======================================================================

def save_pcm_as_wav(pcm_bytes: bytes, output_path: str):
    """Saves raw PCM bytes from Gemini to a WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)

def generate_tts_voiceover(text: str, voice: str = "Puck") -> str:
    """Generate TTS audio for the voiceover using Gemini 2.0 Flash and save to WAV."""
    api_key = os.getenv("GOOGLE_API_KEY")
    gemini_client = genai.Client(api_key=api_key) if api_key else None
    
    if not gemini_client:
        logger.error("GOOGLE_API_KEY is not set. Cannot generate voiceover.")
        return None
        
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
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
        for generated_response in response.candidates[0].content.parts:
            if generated_response.inline_data:
                audio_bytes = generated_response.inline_data.data
                break
                
        if audio_bytes:
            temp_wav = tempfile.mktemp(suffix=".wav")
            save_pcm_as_wav(audio_bytes, temp_wav)
            return temp_wav
        else:
            logger.error("Failed to get audio data from Gemini.")
            return None
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None

def download_vosk_model(model_path="vosk-model-small-en-us-0.15"):
    """Downloads the lightweight Vosk model if it doesn't exist."""
    import urllib.request
    import zipfile
    
    if not os.path.exists(model_path):
        url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
        zip_path = "vosk_model.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
    return model_path

def generate_transcription_timestamps(audio_path: str):
    """Uses Local Vosk to get word-level timestamps (100% Free)."""
    import json
    from vosk import Model, KaldiRecognizer
    
    try:
        model_dir = download_vosk_model()
        model = Model(model_dir)
        wf = wave.open(audio_path, "rb")
        
        if wf.getnchannels() != 1:
            logger.warning("Audio must be Mono for Vosk.")
            return []
            
        rec = KaldiRecognizer(model, wf.getframerate())
        rec.SetWords(True)

        results = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if 'result' in res:
                    results.extend(res['result'])
                    
        res = json.loads(rec.FinalResult())
        if 'result' in res:
            results.extend(res['result'])

        chunks = []
        current_chunk = []
        for w in results:
            current_chunk.append(w)
            if len(current_chunk) >= 3:
                text = " ".join([c['word'].strip() for c in current_chunk]).upper()
                start = current_chunk[0]['start']
                end = current_chunk[-1]['end']
                chunks.append({'text': text, 'start': start, 'end': end})
                current_chunk = []
        
        if current_chunk:
            text = " ".join([c['word'].strip() for c in current_chunk]).upper()
            start = current_chunk[0]['start']
            end = current_chunk[-1]['end']
            chunks.append({'text': text, 'start': start, 'end': end})
            
        return chunks
    except Exception as e:
        logger.error(f"Local Transcription failed: {e}")
        return []

def create_caption_clip(text: str, start: float, end: float, video_size: tuple):
    """Creates a transparent ImageClip with styled text (stroke + color)."""
    from moviepy import ImageClip
    w, h = video_size
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arialbd.ttf", 90)
    except IOError:
        font = ImageFont.load_default()
    
    try:
        bbox = d.textbbox((0,0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = d.textsize(text, font=font)
    
    x = (w - tw) / 2
    y = h * 0.75 
    
    stroke_color = (0, 0, 0, 255)
    stroke_width = 5
    for dx in range(-stroke_width, stroke_width+1, 2):
        for dy in range(-stroke_width, stroke_width+1, 2):
            d.text((x+dx, y+dy), text, font=font, fill=stroke_color)
            
    main_color = (255, 235, 59, 255) # Yellow
    d.text((x, y), text, font=font, fill=main_color)
    
    img_path = tempfile.mktemp(suffix=".png")
    img.save(img_path)
    
    clip = ImageClip(img_path).with_start(start).with_end(end)
    return clip, img_path

# ======================================================================
# MAIN VIDEO GENERATOR NODE
# ======================================================================

def video_generator_node(state: State) -> dict:
    """Generates a stock video short by writing a voiceover script, generating TTS, fetching Pexels clips, and merging them with dynamic captions."""
    from langchain_openai import ChatOpenAI
    
    _emit(_job(state), "video", "started", "Structuring voiceover script for Short...")
    logger.info("🎬 GENERATING STOCK VIDEO ---")
    
    # 1. Setup
    topic = state.get("topic", "Unknown")
    blog_content = state.get("markdown", "")
    
    if not blog_content:
        logger.warning("No blog content found. Skipping video generation.")
        _emit(_job(state), "video", "error", "Skipped: Requires generated blog content.")
        return {"video_path": None}
        
    video_dir = Path("generated_videos")
    video_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.mkdtemp()
    
    # 2. Write Voiceover Script
    _emit(_job(state), "video", "working", "Generating voiceover script...")
    text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"Create a voiceover script for the following blog post.\n\nBLOG POST:\n{blog_content[:4000]}"
    
    response = text_llm.invoke([
        SystemMessage(content=VOICEOVER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    voiceover_script = response.content.strip()
    
    # 3. Generate Audio
    _emit(_job(state), "video", "working", "Synthesizing voiceover audio with Gemini...")
    audio_path = generate_tts_voiceover(voiceover_script, voice="Puck")
    
    if not audio_path:
        logger.error("Audio generation failed.")
        _emit(_job(state), "video", "error", "Audio generation failed.")
        return {"video_path": None}
        
    try:
        from moviepy import AudioFileClip
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return {"video_path": None}
        
    # 4. Generate Captions
    _emit(_job(state), "video", "working", "Transcribing audio for captions (Vosk)...")
    caption_chunks = generate_transcription_timestamps(audio_path)
    
    # 5. Plan video scenes
    _emit(_job(state), "video", "working", "Planning stock video background...")
    planner = llm.with_structured_output(VideoScenePlan)
    plan = planner.invoke([
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
            
        # Stop fetching early if we predict we have enough video time (~15s per clip)
        if len(downloaded_clips) * 15 > audio_duration + 10:
            logger.info("   ✅ Gathered enough footage for the duration.")
            break
            
    # Fallback to a generic query if none worked
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
    _emit(_job(state), "video", "working", "Stitching footage and overlaying audio & captions...")
    logger.info("   ✂️ Editing video with moviepy...")
    
    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips, CompositeVideoClip
        import moviepy.video.fx as vfx
    except ImportError:
        logger.error("moviepy not installed. Skipping video generation.")
        _emit(_job(state), "video", "error", "Missing moviepy dependency.")
        return {"video_path": None}
        
    try:
        # Load video clips
        video_clips = []
        for p in downloaded_clips:
            try:
                clip = VideoFileClip(p)
                # Ensure clips are sized and we take a 10s chunk max
                chunk_length = min(10, clip.duration)
                clip = clip.subclipped(0, chunk_length).resized(width=1080)
                video_clips.append(clip)
            except Exception as e:
                logger.warning(f"Failed to load clip {p}: {e}")
                
        if not video_clips:
            raise Exception("No valid video clips could be loaded by moviepy.")
            
        # Concatenate
        try:
            final_video = concatenate_videoclips(video_clips, method="compose")
        except Exception:
            # Fallback simple concat if compose fails due to fps mismatches
            final_video = concatenate_videoclips(video_clips)
            
        # Match audio duration
        if final_video.duration < audio_duration:
            # Loop video to match audio length
            final_video = final_video.with_effects([vfx.Loop(duration=audio_duration)])
        else:
            # Trim video
            final_video = final_video.subclipped(0, audio_duration)
            
        # Apply Captions
        generated_pngs = []
        if caption_chunks:
            logger.info("   ✍️ Applying dynamic captions to video...")
            video_size = final_video.size
            caption_clips = []
            
            for chunk in caption_chunks:
                cap_clip, png_path = create_caption_clip(chunk['text'], chunk['start'], chunk['end'] + 0.1, video_size)
                caption_clips.append(cap_clip)
                generated_pngs.append(png_path)
                
            final_video = CompositeVideoClip([final_video] + caption_clips)
            
        # Overlay Audio
        final_video = final_video.with_audio(audio_clip)
        
        # Export
        output_file = str(video_dir / f"short_{timestamp}.mp4")
        logger.info(f"   🎬 Exporting to {output_file}...")
        
        final_video.write_videofile(
            output_file, 
            fps=30, 
            codec="libx264", 
            audio_codec="aac",
            preset="ultrafast", # Faster for script generation
            logger=None # Disable verbose output
        )
        
        # Cleanup
        for clip in video_clips: clip.close()
        audio_clip.close()
        final_video.close()
        
        for png in generated_pngs:
            try: os.remove(png)
            except: pass
            
        try: os.remove(audio_path)
        except: pass
        
        logger.info("   ✅ Video Generation Complete!")
        _emit(_job(state), "video", "completed", "Stock video generated successfully.")
        return {"video_path": output_file}
        
    except Exception as e:
        logger.error(f"Video editing failed: {e}")
        _emit(_job(state), "video", "error", f"Video edit failed: {e}")
        return {"video_path": None}
