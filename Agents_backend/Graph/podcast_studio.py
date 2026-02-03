import os
import json
import numpy as np
from scipy.io import wavfile
from typing import List, Tuple, Dict
from google import genai
from google.genai import types

# Initialize Client
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# Audio Constants
SAMPLE_RATE = 24000 

# ============================================================================
# 1. SCRIPTING (Using Existing Blog)
# ============================================================================

def write_podcast_script(topic: str, blog_content: str) -> List[Tuple[str, str]]:
    """
    Generates a script based STRICTLY on the provided blog content.
    """
    print("   ✍️ Writing Script from Blog Content...")

    prompt = f"""
    You are a podcast producer. Convert this blog post into a lively 2-minute dialogue.
    
    TOPIC: {topic}
    SOURCE MATERIAL:
    {blog_content[:15000]} (Truncated if too long)

    CHARACTERS:
    - Fenrir (Skeptic): Critical, asks hard questions.
    - Kore (Expert): Explains the blog's insights clearly.

    INSTRUCTIONS:
    - Do NOT make up new facts. Use the blog content.
    - Keep it under 2 minutes (approx 300 words).
    - Format as JSON list of objects: {{"speaker": "Name", "text": "Spoken line"}}
    """

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", # Fast model is fine here
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        data = json.loads(response.text)
        return [(item["speaker"], item["text"]) for item in data]
    except Exception as e:
        print(f"Script Error: {e}")
        return []

# ============================================================================
# 2. TTS GENERATION (Voices)
# ============================================================================

def generate_speech_audio(script: List[Tuple[str, str]]) -> np.ndarray:
    """Generates audio using Gemini Voices."""
    print("   🗣️ Synthesizing Voices...")
    full_audio = []
    
    # Map: 'SpeakerName' -> 'GeminiVoiceName'
    voice_map = {"Fenrir": "Fenrir", "Kore": "Kore"}

    for speaker, text in script:
        voice_name = voice_map.get(speaker, "Fenrir")
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
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
                    audio_float = np.frombuffer(part.inline_data.data, dtype=np.int16).astype(np.float32) / 32768.0
                    full_audio.append(audio_float)
                    full_audio.append(np.zeros(int(SAMPLE_RATE * 0.4))) # Pause
        except Exception as e:
            print(f"TTS Error: {e}")

    return np.concatenate(full_audio) if full_audio else np.zeros(SAMPLE_RATE)

# ============================================================================
# 3. AUDIO ENGINEERING (Music & Mix)
# ============================================================================

def generate_lofi_jingle(duration_sec: int) -> np.ndarray:
    """Synthesizes Lo-Fi Backing Track (Sine Waves + Noise)."""
    t = np.linspace(0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False)
    output = np.zeros_like(t)
    
    # Chords: C Major -> A Minor
    chords = [[261.63, 329.63, 392.00], [220.00, 261.63, 329.63]]
    
    for i, time_point in enumerate(t):
        chord = chords[int((time_point % 4) // 2)]
        sample = sum(np.sin(2 * np.pi * f * time_point) for f in chord)
        envelope = np.exp(-2 * (time_point % 2)) # Decay per chord
        output[i] = sample * envelope * 0.05 # Low volume

    noise = np.random.normal(0, 0.002, output.shape) # Vinyl crackle
    return (output + noise).astype(np.float32)

def mix_podcast(speech: np.ndarray, music: np.ndarray) -> np.ndarray:
    """Mixes Speech + Music with Ducking."""
    print("   🎚️ Mixing Audio...")
    
    total_len = len(speech) + (4 * SAMPLE_RATE) # +4s padding
    repeats = int(np.ceil(total_len / len(music)))
    music_long = np.tile(music, repeats)[:total_len]
    
    speech_padded = np.zeros(total_len, dtype=np.float32)
    start_idx = 2 * SAMPLE_RATE
    speech_padded[start_idx:start_idx+len(speech)] = speech

    # Automation curve
    automation = np.ones(total_len, dtype=np.float32) * 0.15 # Quiet background
    automation[:start_idx] = 0.5 # Loud Intro
    automation[start_idx+len(speech):] = 0.5 # Loud Outro
    
    return np.clip(speech_padded + (music_long * automation), -1.0, 1.0)

# ============================================================================
# 4. NODE ENTRY POINT
# ============================================================================

def podcast_node(state: Dict) -> Dict:
    """LangGraph Node: Takes 'final' (blog) and produces 'audio_path'."""
    topic = state.get("topic", "podcast")
    blog_content = state.get("final", "")
    
    if not blog_content:
        print("   ⚠️ No blog content found. Skipping podcast.")
        return {"audio_path": None}

    # 1. Script (from Blog)
    script = write_podcast_script(topic, blog_content)
    if not script: return {"audio_path": None}
    
    # 2. Audio Generation
    speech_audio = generate_speech_audio(script)
    music_audio = generate_lofi_jingle(duration_sec=8) # 8s loop
    
    # 3. Mixing
    final_mix = mix_podcast(speech_audio, music_audio)
    
    # 4. Save
    os.makedirs("generated_podcasts", exist_ok=True)
    filename = f"podcast_{topic.replace(' ', '_')[:30]}.wav"
    output_path = os.path.join("generated_podcasts", filename)
    
    wavfile.write(output_path, SAMPLE_RATE, (final_mix * 32767).astype(np.int16))
    
    print(f"   ✅ Podcast Produced: {output_path}")
    return {"audio_path": output_path}