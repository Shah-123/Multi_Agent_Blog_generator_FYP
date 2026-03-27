import os
import sys
import tempfile
import wave
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import internal modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types
from Graph.agents.utils import llm
from Graph.agents.video import VideoScenePlan, VIDEO_PLAN_SYSTEM, fetch_pexels_video

# Initialize Gemini Client (Audio)
api_key = os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None

if not gemini_client:
    print("❌ Error: GOOGLE_API_KEY is not set in your .env file.")
    sys.exit(1)

# ============================================================================
# VOICEOVER SCRIPT GENERATION
# ============================================================================

VOICEOVER_SYSTEM_PROMPT = """You are a professional YouTube Shorts / TikTok scriptwriter.
Given the blog post, write a highly engaging, fast-paced voiceover script to be read by a single narrator.
The script should be around 45 to 60 seconds long when spoken (about 120-150 words).
Ensure:
- Strong hook at the beginning
- Clear, punchy educational points
- A call to action at the end
- DO NOT INCLUDE ANY SPEAKER LABELS LIKE "NARRATOR:" OR "VOICEOVER:". Just output the raw text to be spoken.
"""

def save_pcm_as_wav(pcm_bytes: bytes, output_path: str):
    """Saves raw PCM bytes from Gemini to a WAV file."""
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)

def generate_tts_voiceover(text: str, voice: str = "Puck") -> str:
    """Generate TTS audio for the voiceover using Gemini 2.5 Pro and save to WAV."""
    print(f"   🔊 Generating audio with Gemini-2.5-Pro (Voice: {voice})...")
    try:
        # Note: Gemini 2.0+ Flash supports native Audio output using the 'v1alpha' capability or standard API.
        # However, the SDK might strictly enforce text for 2.5-pro right now. We use 2.0-flash as it's
        # the stable multimodal generative model.
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
        
        # Extract audio bytes
        audio_bytes = None
        for generated_response in response.candidates[0].content.parts:
            if generated_response.inline_data:
                audio_bytes = generated_response.inline_data.data
                break
                
        if audio_bytes:
            temp_wav = tempfile.mktemp(suffix=".wav")
            save_pcm_as_wav(audio_bytes, temp_wav)
            print("   ✅ Audio generated successfully.")
            return temp_wav
        else:
            print("   ❌ Failed to get audio data from Gemini.")
            return None
    except Exception as e:
        print(f"   ⚠️ TTS generation failed: {e}")
        return None

# ======================================================================
# STANDALONE VOICEOVER + VIDEO GENERATOR
# ======================================================================

def generate_voiceover_video_from_blog(blog_path: str):
    """
    Reads a blog, writes a short voiceover, generates audio via Gemini,
    fetches Pexels stock video, and combines them using MoviePy.
    """
    blog_path_obj = Path(blog_path).absolute()
    if not blog_path_obj.exists():
        print(f"❌ Error: Could not find blog file at {blog_path_obj}")
        return

    blog_path_str = str(blog_path_obj)

    print("="*60)
    print("🎬 STANDALONE VOICEOVER + VIDEO GENERATOR")
    print("="*60)

    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_videoclips
        import moviepy.video.fx as vfx
    except ImportError as e:
        print(f"❌ Error: moviepy failed to import: {e}")
        return

    # 1. Read the blog content
    print(f"📖 Reading blog content from: {os.path.basename(blog_path_str)}")
    with open(blog_path_str, 'r', encoding='utf-8') as f:
         content = f.read()
         
    topic = os.path.basename(blog_path_str).replace(".md", "").replace("_", " ")

    # 2. Write Voiceover Script
    print("\n🧠 Writing Voiceover Script using LLM...")
    text_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"Create a voiceover script for the following blog post.\n\nBLOG POST:\n{content[:4000]}"
    
    response = text_llm.invoke([
        SystemMessage(content=VOICEOVER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    script = response.content.strip()
    print(f"📝 Script generation complete:\n---\n{script}\n---")
    
    # 3. Generate Audio
    print("\n🎧 Generating Voiceover Audio...")
    audio_path = generate_tts_voiceover(script, voice="Puck")
    if not audio_path:
        print("❌ Audio generation failed. Aborting.")
        return
        
    # Load audio to get its duration
    try:
        final_audio = AudioFileClip(audio_path)
        audio_duration = final_audio.duration
        print(f"   ⏱️ Audio duration: {audio_duration:.2f} seconds")
    except Exception as e:
        print(f"❌ Failed to load generated audio: {e}")
        return

    # 4. Plan Video Scenes based on Voiceover
    print("\n🧠 Planning video scenes from voiceover...")
    planner = llm.with_structured_output(VideoScenePlan)
    plan = planner.invoke([
        SystemMessage(content=VIDEO_PLAN_SYSTEM),
        HumanMessage(content=f"Topic: {topic}\n\nVoiceover Script:\n{script}")
    ])

    queries = plan.keywords if plan.keywords else [topic, "abstract background", "technology"]
    print(f"🎥 Generated Search Queries for Pexels: {queries}")

    # 5. Fetch from Pexels
    temp_dir = tempfile.mkdtemp()
    print(f"\n📥 Downloading clips to temporary directory: {temp_dir}")
    downloaded_clips = []
    
    for i, q in enumerate(queries):
        print(f"   🔍 Searching Pexels for: '{q}'...")
        clip_path = fetch_pexels_video(q, temp_dir, i)
        if clip_path:
            print(f"   ✅ Downloaded: {os.path.basename(clip_path)}")
            downloaded_clips.append(clip_path)
            
        # Stop fetching early if we predict we have enough video time. Default clips are ~15 seconds.
        # So 15 * len(downloaded_clips) > audio_duration
        if len(downloaded_clips) * 15 > audio_duration + 10:
            print("   ✅ Gathered enough footage for the duration.")
            break
            
    if not downloaded_clips:
        print("❌ Error: Failed to download any stock footage.")
        return

    # 6. Stitch with Moviepy
    print("\n✂️ Stitching Video & Audio together...")
    video_clips = []
    current_duration = 0
    
    for p in downloaded_clips:
        try:
            clip = VideoFileClip(p)
            
            # If we already have enough duration, we don't strictly *need* more clips, 
            # but we can add them truncated. Let's just grab 5-10s chunks of each.
            chunk_length = min(10, clip.duration)
            clip = clip.subclipped(0, chunk_length).resized(width=1080)
            
            video_clips.append(clip)
            current_duration += clip.duration
        except Exception as e:
            print(f"   ⚠️ Failed to load clip {p}: {e}")
            
    if not video_clips:
        print("❌ Error: No valid video clips loaded.")
        return
        
    try:
        print("   🔄 Processing video frames...")
        final_video = concatenate_videoclips(video_clips)
        
        # Match audio duration exactly
        if final_video.duration < audio_duration:
            print("   🔂 Looping video to match audio length...")
            final_video = final_video.with_effects([vfx.Loop(duration=audio_duration)])
        else:
            final_video = final_video.subclipped(0, audio_duration)
            
        # Set the voiceover track!
        final_video = final_video.with_audio(final_audio)
        
        # Save output
        output_file = "test_voiceover_short.mp4"
        
        print(f"\n🎬 Exporting Final Render to: {output_file}")
        print("   (This step might take a few moments...)")
        final_video.write_videofile(
            output_file, 
            fps=30, 
            codec="libx264", 
            audio_codec="aac",
            preset="ultrafast" # Keep it fast for testing
        )
        
        # Cleanup
        for clip in video_clips: clip.close()
        final_audio.close()
        final_video.close()
        
        try:
            os.remove(audio_path)
        except:
            pass
            
        print(f"\n✨ SUCCESS! Short video saved as {output_file}")
        
    except Exception as e:
        print(f"❌ Video editing failed: {e}")

if __name__ == "__main__":
    print("\n")
    print("Provide the absolute path to a generated blog markdown file.")
    print("Example: C:/Users/SHAHKAR/Multi_Agent_Blog_generator_FYP/Agents_backend/blogs/ai_in_healthcare/content/blog.md")
    target_blog = input("> ").strip()
    
    if target_blog:
        target_blog = target_blog.strip('"\'') 
        generate_voiceover_video_from_blog(target_blog)


