import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import internal modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from google import genai
from google.genai import types

# Initialize Gemini Client (Audio)
api_key = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=api_key) if api_key else None

if not gemini_client:
    print("❌ Error: GEMINI_API_KEY is not set in your .env file.")
    sys.exit(1)

# ============================================================================
# PODCAST SCRIPT GENERATION
# ============================================================================

PODCAST_SYSTEM_PROMPT = """You are a podcast script writer creating engaging conversational content based on a blog post.

Format your script EXACTLY like this:

HOST: [Opening statement or question]

GUEST: [Response with insights]

HOST: [Follow-up question or transition]

GUEST: [Detailed explanation]

... continue this pattern ...

HOST: [Closing remarks]

RULES:
- Use "HOST:" and "GUEST:" labels consistently (CRITICAL)
- Keep each speaker turn to 2-4 sentences max
- Make it conversational and natural
- Include pauses like "hmm", "you know", "right"
- Total length: 6-10 exchanges
- Educational but entertaining tone
"""

def split_script_into_segments(script: str, max_chars: int = 3500) -> list:
    """Split script into segments at natural speaker boundaries."""
    parts = re.split(r'(HOST:|GUEST:)', script)
    
    segments = []
    current_segment = ""
    
    for i in range(1, len(parts), 2):
        speaker_label = parts[i]      # e.g., "HOST:"
        content = parts[i+1].strip()  # The text following it
        
        turn = f"{speaker_label} {content}\n\n"
        
        if len(current_segment) + len(turn) > max_chars and current_segment:
            segments.append(current_segment.strip())
            current_segment = turn
        else:
            current_segment += turn
    
    if current_segment:
        segments.append(current_segment.strip())
    
    return segments if segments else [script]

def get_voice_for_speaker(text: str) -> str:
    # Gemini voices: Puck (male), Aoede (female), Charon (calm male), Kore (friendly female), Fenrir, Leto
    if "HOST:" in text[:10]:
        return "Puck" 
    elif "GUEST:" in text[:10]:
        return "Aoede"   
    return "Charon"

def generate_tts_segment(text: str, voice: str = "Puck") -> bytes:
    """Generate TTS audio for a text segment using Gemini 2.5 Pro."""
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-pro',
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
        # The SDK returns audio data natively now
        for generated_response in response.candidates[0].content.parts:
            if generated_response.inline_data:
                return generated_response.inline_data.data
        return None
    except Exception as e:
        print(f"   ⚠️ TTS generation failed: {e}")
        return None

def save_pcm_as_wav(pcm_bytes: bytes, output_path: str):
    import wave
    # Gemini outputs 24kHz, 1-channel (mono), 16-bit PCM by default
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2) # 16-bit = 2 bytes
        wf.setframerate(24000)
        wf.writeframes(pcm_bytes)

def combine_wav_files(audio_files: list, output_path: str) -> str:
    """Combines WAV files by extracting their frames and writing a single merged WAV."""
    if not audio_files:
        return None

    import wave
    print(f"   🎚️ Combining {len(audio_files)} audio segments into WAV...")
    
    try:
        # Get params from first file
        with wave.open(audio_files[0], 'rb') as w:
            params = w.getparams()
            
        with wave.open(output_path, 'wb') as outfile:
            outfile.setparams(params)
            for f in audio_files:
                with wave.open(f, 'rb') as infile:
                    outfile.writeframes(infile.readframes(infile.getnframes()))
                    
        print("   ✅ Combined using wave module directly")
        return output_path
    except Exception as e:
        print(f"   ❌ Audio combination failed completely: {e}")
        return None

# ======================================================================
# STANDALONE AUDIO GENERATOR 
# ======================================================================

def generate_podcast_from_blog(blog_path: str):
    """
    Reads a completed blog post, generates a podcast script via LLM,
    and synthesizes MP3 TTS chunks to form a single podcast MP3.
    """
    blog_path_obj = Path(blog_path).absolute()
    if not blog_path_obj.exists():
        print(f"❌ Error: Could not find blog file at {blog_path_obj}")
        return

    blog_path_str = str(blog_path_obj)

    print("="*60)
    print("🎙️ STANDALONE PODCAST AUDIO GENERATOR (OPENAI TTS)")
    print("="*60)

    # 1. Read the blog content
    print(f"📖 Reading blog content from: {os.path.basename(blog_path_str)}")
    with open(blog_path_str, 'r', encoding='utf-8') as f:
         content = f.read()

    # 2. Plan Podcast Script
    print("🧠 Writing Podcast Script using LLM...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    prompt = f"Create a conversational podcast script for the following blog post.\n\nBLOG POST:\n{content[:4000]}"
    
    response = llm.invoke([
        SystemMessage(content=PODCAST_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ])
    
    script = response.content
    
    if "HOST:" not in script:
        script = f"HOST: Welcome to the show!\n\nGUEST: Thanks for having me.\n\n" + script
        
    print(f"📝 Script generation complete. ({len(script)} characters)")
    
    # 3. Process Audio
    print(f"🎧 Generating Audio via OPENAI TTS...")
    segments = split_script_into_segments(script)
    print(f"   📊 Split into {len(segments)} narrative segments.")
    
    audio_files = []
    output_dir = Path("temp_podcast_chunks")
    output_dir.mkdir(exist_ok=True)
    
    for i, segment in enumerate(segments):
        voice = get_voice_for_speaker(segment)
        print(f"   🔊 Requesting chunk {i+1}/{len(segments)} (Voice: {voice})...")
        audio_bytes = generate_tts_segment(segment, voice=voice)
        
        if audio_bytes:
            seg_path = output_dir / f"chunk_{i:03d}.wav"
            save_pcm_as_wav(audio_bytes, str(seg_path))
            audio_files.append(str(seg_path))
        else:
            print(f"   ❌ Failed chunk {i+1}")

    if not audio_files:
        print("❌ Error: Failed to generate any TTS audio segments.")
        return
        
    # 4. Combine Audio
    print("\n✂️ Stitching audio together...")
    output_file = "test_podcast.wav"
    
    if len(audio_files) == 1:
        import shutil
        shutil.copy(audio_files[0], output_file)
        print("   ✅ Single chunk saved directly.")
    else:
        combine_wav_files(audio_files, output_file)
        
    # Cleanup Segments
    for f in audio_files:
        try:
            os.remove(f)
        except Exception: 
            pass
    try:
        os.rmdir(output_dir)
    except Exception:
        pass
        
    print(f"\n✨ SUCCESS! Podcast saved as {output_file}")


if __name__ == "__main__":
    print("\n")
    print("Provide the absolute path to a generated blog markdown file.")
    print("Example: C:/Users/SHAHKAR/Multi_Agent_Blog_generator_FYP/Agents_backend/blogs/ai_in_healthcare/content/blog.md")
    target_blog = input("> ").strip()
    
    if target_blog:
        # Clean up quotes if dragged-and-dropped in terminal
        target_blog = target_blog.strip('"\'') 
        generate_podcast_from_blog(target_blog)
