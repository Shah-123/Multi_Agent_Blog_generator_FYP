import os
import json
import numpy as np
from scipy.io import wavfile
from typing import List, Tuple, Dict, Optional
from pathlib import Path

# Initialize Client only if API key exists
client = None
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except ImportError:
        print("⚠️ Google GenAI not installed. Skipping podcast generation.")
        client = None

# Audio Constants
SAMPLE_RATE = 24000 

# ============================================================================
# 1. SCRIPTING (Using Existing Blog) - FIXED PROMPT
# ============================================================================

def write_podcast_script(topic: str, blog_content: str) -> List[Tuple[str, str]]:
    """
    Generates a script based STRICTLY on the provided blog content.
    """
    print("   ✍️ Writing Script from Blog Content...")
    
    if not client:
        print("   ⚠️ Google API not available. Skipping script generation.")
        return []

    prompt = f"""
    Convert this blog post into a lively 2-minute podcast dialogue between two hosts.
    
    TOPIC: {topic}
    
    BLOG CONTENT:
    {blog_content[:8000]}...

    CHARACTERS:
    - Host A (Expert): Explains concepts clearly, enthusiastic
    - Host B (Curious Listener): Asks questions, seeks clarification

    REQUIREMENTS:
    1. Use ONLY facts from the blog content above
    2. Keep it conversational and engaging
    3. 2-minute duration (approx 300 words total)
    4. Format as JSON array of objects with "speaker" and "text"

    OUTPUT FORMAT:
    [
      {{"speaker": "Host A", "text": "Welcome to today's show..."}},
      {{"speaker": "Host B", "text": "That's fascinating! Tell me more..."}}
    ]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro-preview",  # Use available model
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Extract JSON from response
        content = response.text.strip()
        
        # Clean up markdown code blocks if present
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove ```
        content = content.strip()
        
        data = json.loads(content)
        
        # Validate format
        if isinstance(data, list) and len(data) > 0:
            script = []
            for item in data:
                speaker = item.get("speaker", "Host A")
                text = item.get("text", "")
                if text:  # Only add if there's actual text
                    script.append((speaker, text))
            return script
        else:
            print("   ❌ Invalid script format received.")
            return []
            
    except json.JSONDecodeError as e:
        print(f"   ❌ Failed to parse script JSON: {e}")
        print(f"   Raw response: {content[:200]}...")
        return []
    except Exception as e:
        print(f"   ❌ Script generation error: {e}")
        return []

# ============================================================================
# 2. TTS GENERATION (Voices) - FIXED WITH FALLBACK
# ============================================================================

def generate_speech_audio(script: List[Tuple[str, str]], output_folder: str = None) -> Optional[np.ndarray]:
    """Generates audio using Gemini Voices with fallback."""
    print("   🗣️ Synthesizing Voices...")
    
    if not client or not script:
        print("   ⚠️ No client or script. Skipping TTS.")
        return None
    
    full_audio = []
    
    # Check available voices (this is critical - not all voices work!)
    # Try different voice combinations
    voice_combinations = [
        {"Host A": "Puck", "Host B": "Fable"},
        {"Host A": "Nova", "Host B": "Fable"},
        {"Host A": "Ash", "Host B": "Coral"},
    ]
    
    # Try each combination until one works
    for voice_map in voice_combinations:
        try:
            for speaker, text in script[:2]:  # Test with first 2 lines
                voice_name = voice_map.get(speaker, "Fable")
                response = client.models.generate_content(
                    model="gemini-2.5-pro-preview-tts",  # Try this model for TTS
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        audio_config=types.AudioConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name
                                )
                            )
                        )
                    )
                )
                
                # If we get here, this voice combination works
                print(f"   ✅ Using voices: {voice_map}")
                break
            break
        except Exception as e:
            print(f"   ⚠️ Voice combination failed: {voice_map}")
            continue
    
    # Now generate full script with working voices
    try:
        for speaker, text in script:
            voice_name = voice_map.get(speaker, "Fable")
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    audio_config=types.AudioConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=voice_name
                            )
                        )
                    )
                )
            )
            
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    # Convert audio bytes to numpy array
                    audio_data = part.inline_data.data
                    audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    
                    full_audio.append(audio_float)
                    full_audio.append(np.zeros(int(SAMPLE_RATE * 0.3)))  # 0.3s pause
                    
        if full_audio:
            return np.concatenate(full_audio)
        else:
            return None
            
    except Exception as e:
        print(f"   ❌ TTS generation failed: {e}")
        return None

# ============================================================================
# 3. SIMPLE MUSIC GENERATION (No Complex Synthesis)
# ============================================================================

def add_background_music(speech_audio: np.ndarray) -> np.ndarray:
    """Add simple background music/ambiance."""
    print("   🎵 Adding background ambiance...")
    
    if speech_audio is None:
        return None
    
    # Create simple background (brown noise + gentle sine wave)
    duration = len(speech_audio) / SAMPLE_RATE
    t = np.linspace(0, duration, len(speech_audio), endpoint=False)
    
    # Brown noise (softer than white noise)
    brown_noise = np.cumsum(np.random.randn(len(t)))
    brown_noise = brown_noise / np.max(np.abs(brown_noise)) * 0.03  # Very quiet
    
    # Gentle low-frequency sine wave
    sine_wave = np.sin(2 * np.pi * 60 * t) * 0.01  # Very soft 60Hz
    
    background = brown_noise + sine_wave
    
    # Mix speech with background (ducking)
    # Reduce background when speech is present
    speech_envelope = np.abs(speech_audio)
    speech_envelope = np.clip(speech_envelope * 10, 0, 1)  # Normalize
    
    # When speech is loud, background is quieter
    ducked_background = background * (1 - speech_envelope * 0.7)
    
    # Mix
    mixed = speech_audio + ducked_background
    mixed = np.clip(mixed, -1.0, 1.0)
    
    return mixed

# ============================================================================
# 4. NODE ENTRY POINT - INTEGRATED WITH YOUR PROJECT
# ============================================================================

def podcast_node(state: Dict) -> Dict:
    """LangGraph Node: Takes blog content and produces audio."""
    print("--- 🎙️ GENERATING PODCAST ---")
    
    topic = state.get("topic", "podcast")
    blog_content = state.get("final", "")
    blog_folder = state.get("blog_folder", ".")
    
    if not blog_content:
        print("   ⚠️ No blog content found. Skipping podcast.")
        return {"audio_path": None}
    
    # 1. Generate script from blog
    script = write_podcast_script(topic, blog_content)
    if not script:
        print("   ⚠️ Failed to generate script. Skipping podcast.")
        return {"audio_path": None}
    
    print(f"   📝 Generated {len(script)} dialogue lines")
    
    # 2. Generate speech audio
    speech_audio = generate_speech_audio(script, blog_folder)
    if speech_audio is None:
        print("   ⚠️ Failed to generate speech. Creating text-only summary.")
        
        # Fallback: Create a simple text file with the script
        audio_dir = Path(f"{blog_folder}/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        script_file = audio_dir / f"podcast_script_{topic[:30].replace(' ', '_')}.txt"
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(f"PODCAST SCRIPT: {topic}\n")
            f.write("=" * 50 + "\n\n")
            for speaker, text in script:
                f.write(f"{speaker}:\n{text}\n\n")
        
        print(f"   ✅ Saved podcast script to: {script_file}")
        return {"audio_path": None, "script_path": str(script_file)}
    
    # 3. Add background music
    final_audio = add_background_music(speech_audio)
    
    # 4. Save to organized folder
    audio_dir = Path(f"{blog_folder}/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"podcast_{topic[:30].replace(' ', '_')}.wav"
    output_path = audio_dir / filename
    
    # Convert to int16 for WAV format
    audio_int16 = (final_audio * 32767).astype(np.int16)
    wavfile.write(output_path, SAMPLE_RATE, audio_int16)
    
    print(f"   ✅ Podcast saved to: {output_path}")
    
    # Also save script as text
    script_file = audio_dir / f"script_{topic[:30].replace(' ', '_')}.txt"
    with open(script_file, 'w', encoding='utf-8') as f:
        f.write(f"PODCAST SCRIPT: {topic}\n")
        f.write("=" * 50 + "\n\n")
        for speaker, text in script:
            f.write(f"{speaker}:\n{text}\n\n")
    
    return {
        "audio_path": str(output_path),
        "script_path": str(script_file)
    }