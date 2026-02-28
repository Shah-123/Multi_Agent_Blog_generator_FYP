import os
import re
from typing import Optional
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage

from Graph.state import State, GlobalImagePlan
from Graph.templates import DECIDE_IMAGES_SYSTEM
from .utils import logger, llm, _job, _emit, _safe_slug

def decide_images(state: State) -> dict:
    _emit(_job(state), "images", "started", "Planning image placement...")
    logger.info("🖼️ PLANNING IMAGES ---")
    planner = llm.with_structured_output(GlobalImagePlan)
    
    image_plan = planner.invoke([
        SystemMessage(content=DECIDE_IMAGES_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Current Blog Content:\n{state['merged_md']}" 
        )),
    ])

    _emit(_job(state), "images", "working", f"Planned {len(image_plan.images)} images", {"count": len(image_plan.images)})
    return {
        "image_specs": [img.model_dump() for img in image_plan.images],
    }

def _generate_image_bytes_google(prompt: str) -> Optional[bytes]:
    """Generates image using Google GenAI (Gemini)."""
    try:
        from google import genai
        from google.genai import types
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: 
            return None

        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model="gemini-2.5-flash-image", 
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
        return None
    except ImportError:
        logger.warning("Google GenAI library not installed.")
        return None
    except Exception as e:
        logger.warning(f"Image Gen Error: {e}")
        return None

def generate_and_place_images(state: State) -> dict:
    """Generates images, replaces placeholders, and returns final text."""
    _emit(_job(state), "images", "working", "Generating AI images...")
    logger.info("🎨 GENERATING IMAGES & SAVING ---")
    
    plan = state["plan"]
    final_md = state.get("merged_md", "")
    image_specs = state.get("image_specs", [])
    
    base_path = state.get("blog_folder", ".")
    assets_path = f"{base_path}/assets/images"
    
    if os.getenv("GOOGLE_API_KEY") and image_specs:
        logger.info(f"Attempting to generate {len(image_specs)} images...")
        
        Path(assets_path).mkdir(parents=True, exist_ok=True)

        for img in image_specs:
            img_bytes = _generate_image_bytes_google(img["prompt"])
            
            if img_bytes:
                img_filename = _safe_slug(img["filename"])
                if not img_filename.endswith(".png"): img_filename += ".png"
                
                full_path = Path(f"{assets_path}/{img_filename}")
                full_path.write_bytes(img_bytes)
                
                rel_path = f"../assets/images/{img_filename}"
                markdown_image = f"\n\n![{img['alt']}]({rel_path})\n*Figure: {img['caption']}*\n\n"
                
                target_phrase = img.get("target_paragraph", "").strip()
                if target_phrase:
                    paragraphs = re.split(r'\n\s*\n', final_md)
                    target_words = set(target_phrase.lower().split())
                    
                    best_match_idx = -1
                    best_score = 0
                    
                    for i, p in enumerate(paragraphs):
                        p_words = set(p.lower().split())
                        overlap = len(target_words.intersection(p_words))
                        if overlap > best_score:
                            best_score = overlap
                            best_match_idx = i
                    
                    if best_match_idx >= 0 and best_score >= min(3, len(target_words) // 3):
                        paragraphs.insert(best_match_idx + 1, markdown_image.strip())
                        final_md = "\n\n".join(paragraphs)
                    else:
                        logger.warning(f"Could not find target paragraph similar to '{target_phrase}', appending to end.")
                        final_md += "\n" + markdown_image
                else:
                    final_md += "\n" + markdown_image

                logger.info(f"✅ Generated: {img_filename}")
            else:
                logger.error(f"Failed: {img['filename']} (skipping)")
                
    else:
        logger.info("⏭️ Skipped Image Generation (No API Key or no specs)")

    _emit(_job(state), "images", "completed", "Images processed")
    return {"final": final_md}
