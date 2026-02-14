# import os
# import re
# from datetime import datetime
# from pathlib import Path
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from openai import OpenAI

# # Initialize OpenAI Client (Audio)
# # Ensure API key is present
# api_key = os.getenv("OPENAI_API_KEY")
# openai_client = OpenAI(api_key=api_key) if api_key else None

# # ============================================================================
# # PODCAST SCRIPT GENERATION
# # ============================================================================

# PODCAST_SYSTEM_PROMPT = """You are a podcast script writer creating engaging conversational content.

# Format your script EXACTLY like this:

# HOST: [Opening statement or question]

# GUEST: [Response with insights]

# HOST: [Follow-up question or transition]

# GUEST: [Detailed explanation]

# ... continue this pattern ...

# HOST: [Closing remarks]

# RULES:
# - Use "HOST:" and "GUEST:" labels consistently
# - Keep each speaker turn to 2-4 sentences max
# - Make it conversational and natural
# - Include pauses like "hmm", "you know", "right"
# - Total length: 8-12 exchanges
# - Educational but entertaining tone
# """

# def generate_podcast_script(state: dict) -> str:
#     """Generate a conversational podcast script from blog content."""
#     plan = state.get("plan")
#     topic = state.get("topic", "the topic")
    
#     # Fallback if no plan exists
#     if not plan:
#         return f"HOST: Welcome! Today we are discussing {topic}.\nGUEST: It is great to be here."
    
#     # Build context from sections
#     sections_summary = "\n".join([f"- {task.title}" for task in plan.tasks])
    
#     prompt = f"""Create a podcast script discussing: "{plan.blog_title}"

# Key sections to cover:
# {sections_summary}

# Target audience: {plan.audience}
# Tone: {plan.tone}

# Create a 3-5 minute conversational podcast between a HOST and GUEST expert.
# Make it engaging, educational, and natural-sounding."""

#     llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
#     response = llm.invoke([
#         SystemMessage(content=PODCAST_SYSTEM_PROMPT),
#         HumanMessage(content=prompt)
#     ])
    
#     script = response.content
    
#     # Simple validation
#     if "HOST:" not in script:
#         script = f"HOST: Welcome to our show about {topic}!\n\nGUEST: Thanks for having me.\n\nHOST: Let's dive in.\n\n" + script
        
#     return script

# # ============================================================================
# # AUDIO GENERATION UTILS
# # ============================================================================

# def split_script_into_segments(script: str, max_chars: int = 3500) -> list:
#     """
#     Split script into segments at natural speaker boundaries.
#     """
#     # Split keeping delimiters. 
#     # re.split returns [pre_text, delim, post_text, delim, post_text...]
#     parts = re.split(r'(HOST:|GUEST:)', script)
    
#     segments = []
#     current_segment = ""
    
#     # parts[0] is usually empty text before the first label. 
#     # Start iterating from index 1 where the first "HOST:" or "GUEST:" should be.
#     for i in range(1, len(parts), 2):
#         speaker_label = parts[i]      # e.g., "HOST:"
#         content = parts[i+1].strip()  # The text following it
        
#         turn = f"{speaker_label} {content}\n\n"
        
#         if len(current_segment) + len(turn) > max_chars and current_segment:
#             segments.append(current_segment.strip())
#             current_segment = turn
#         else:
#             current_segment += turn
    
#     if current_segment:
#         segments.append(current_segment.strip())
    
#     return segments if segments else [script]

# def get_voice_for_speaker(text: str) -> str:
#     if "HOST:" in text[:10]:
#         return "alloy"  # Male-ish
#     elif "GUEST:" in text[:10]:
#         return "nova"   # Female-ish
#     return "onyx"

# def generate_tts_segment(text: str, voice: str = "alloy") -> bytes:
#     """Generate TTS audio for a text segment."""
#     if not openai_client:
#         print("   ⚠️ OpenAI API Key missing for TTS.")
#         return None
#     try:
#         response = openai_client.audio.speech.create(
#             model="tts-1",
#             voice=voice,
#             input=text,
#             response_format="mp3"
#         )
#         return response.content
#     except Exception as e:
#         print(f"   ⚠️ TTS generation failed: {e}")
#         return None

# def combine_mp3_files(audio_files: list, output_path: str) -> str:
#     """
#     Combines MP3 files. 
#     Tries Pydub first (best quality, requires FFmpeg).
#     Falls back to binary concatenation (risky but often works for same-format MP3s).
#     """
#     if not audio_files:
#         return None

#     print(f"   🎚️ Combining {len(audio_files)} audio segments...")
    
#     # 1. Try Pydub (Requires FFmpeg installed on OS)
#     try:
#         from pydub import AudioSegment
#         combined = AudioSegment.empty()
#         for f in audio_files:
#             combined += AudioSegment.from_mp3(f)
        
#         combined.export(output_path, format="mp3", bitrate="192k")
#         print("   ✅ Combined using Pydub")
#         return output_path
#     except ImportError:
#         print("   ⚠️ Pydub not installed. Using fallback.")
#     except Exception as e:
#         print(f"   ⚠️ Pydub failed (likely missing FFmpeg): {e}. Using binary fallback.")

#     # 2. Fallback: Binary Concatenation (Works if all chunks are same format/bitrate)
#     try:
#         with open(output_path, 'wb') as outfile:
#             for f in audio_files:
#                 with open(f, 'rb') as infile:
#                     outfile.write(infile.read())
#         print("   ✅ Combined using binary concatenation (fallback)")
#         return output_path
#     except Exception as e:
#         print(f"   ❌ Audio combination failed completely: {e}")
#         return None

# # ============================================================================
# # MAIN NODE
# # ============================================================================

# def podcast_node(state: dict) -> dict:
#     """
#     Generate podcast audio from blog content.
#     Returns: {"audio_path": str, "script_path": str}
#     """
#     print("--- 🎙️ PODCAST STATION ---")
    
#     # Setup
#     podcast_dir = Path("generated_podcasts")
#     podcast_dir.mkdir(exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # 1. Generate Script
#     print(f"   ✍️ Writing Podcast Script...")
#     script = generate_podcast_script(state)
#     script_path = podcast_dir / f"script_{timestamp}.txt"
#     script_path.write_text(script, encoding="utf-8")
    
#     # 2. Split
#     segments = split_script_into_segments(script)
#     print(f"   📊 Processing {len(segments)} script segments...")
    
#     # 3. Generate Audio
#     audio_files = []
    
#     for i, segment in enumerate(segments):
#         voice = get_voice_for_speaker(segment)
#         audio_bytes = generate_tts_segment(segment, voice=voice)
        
#         if audio_bytes:
#             seg_path = podcast_dir / f"seg_{timestamp}_{i:03d}.mp3"
#             seg_path.write_bytes(audio_bytes)
#             audio_files.append(str(seg_path))
#             print(f"   ✅ Segment {i+1}/{len(segments)} ({voice})")
#         else:
#             print(f"   ❌ Failed segment {i+1}")

#     if not audio_files:
#         return {"audio_path": None, "script_path": str(script_path)}
    
#     # 4. Combine
#     final_path = podcast_dir / f"podcast_{timestamp}.mp3"
    
#     # If we only have one segment, just rename it
#     if len(audio_files) == 1:
#         os.rename(audio_files[0], final_path)
#         combined_path = str(final_path)
#     else:
#         combined_path = combine_mp3_files(audio_files, str(final_path))
    
#     # Cleanup Segments
#     if combined_path and os.getenv("KEEP_AUDIO_SEGMENTS", "false").lower() == "false":
#         for f in audio_files:
#             try:
#                 if f != combined_path: os.remove(f)
#             except: pass

#     return {
#         "audio_path": combined_path,
#         "script_path": str(script_path)
#     }