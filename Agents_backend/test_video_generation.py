import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path so we can import internal modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from Graph.agents.utils import llm, logger
from Graph.agents.video import VideoScenePlan, VIDEO_PLAN_SYSTEM, fetch_pexels_video

# ======================================================================
# STANDALONE VIDEO GENERATOR (NO AUDIO)
# ======================================================================

def generate_video_from_blog(blog_path: str):
    """
    Reads a completed blog post, plans video scenes, fetches from Pexels,
    and stitches them together into a silent MP4.
    """
    # Resolve path
    blog_path_obj = Path(blog_path).absolute()
    if not blog_path_obj.exists():
        print(f"❌ Error: Could not find blog file at {blog_path_obj}")
        return

    blog_path = str(blog_path_obj)

    print("="*60)
    print("🎬 STANDALONE VIDEO GENERATOR (PEXELS + MOVIEPY)")
    print("="*60)

    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError as e:
        print(f"❌ Error: moviepy failed to import: {e}")
        import traceback
        traceback.print_exc()
        return

    # 1. Read the blog content
    print(f"📖 Reading blog content from: {os.path.basename(blog_path)}")
    with open(blog_path, 'r', encoding='utf-8') as f:
         content = f.read()

    # We use the filename as a proxy for the topic
    topic = os.path.basename(blog_path).replace(".md", "").replace("_", " ")

    # 2. Plan Video Scenes
    print("🧠 Planning video scenes using LLM...")
    planner = llm.with_structured_output(VideoScenePlan)
    
    # We pass the first ~3000 characters of the blog so the LLM gets the context
    plan = planner.invoke([
        SystemMessage(content=VIDEO_PLAN_SYSTEM),
        HumanMessage(content=f"Topic: {topic}\n\nBlog Intro:\n{content[:3000]}")
    ])

    queries = plan.keywords if plan.keywords else [topic, "abstract background", "technology"]
    print(f"🎥 Generated Search Queries: {queries}")

    # 3. Fetch from Pexels
    temp_dir = tempfile.mkdtemp()
    print(f"📥 Downloading clips to temporary directory: {temp_dir}")
    downloaded_clips = []
    
    for i, q in enumerate(queries):
        print(f"   🔍 Searching Pexels for: '{q}'...")
        clip_path = fetch_pexels_video(q, temp_dir, i)
        if clip_path:
            print(f"   ✅ Downloaded: {os.path.basename(clip_path)}")
            downloaded_clips.append(clip_path)
            
    if not downloaded_clips:
        print("❌ Error: Failed to download any stock footage.")
        return

    # 4. Stitch with Moviepy (Silent Video)
    print("\n✂️ Stitching clips together (Silent Video)...")
    video_clips = []
    for p in downloaded_clips:
        try:
            clip = VideoFileClip(p)
            # Default stock videos are often 10-15 seconds. Let's just use the first 5 seconds to keep it snappy.
            # We also ensure the width is 1080 to normalize them.
            clip = clip.subclipped(0, min(5, clip.duration)).resized(width=1080)
            video_clips.append(clip)
        except Exception as e:
            print(f"   ⚠️ Failed to load clip {p}: {e}")
            
    if not video_clips:
        print("❌ Error: No valid video clips loaded.")
        return
        
    try:
        # Concatenate using simple method 
        print("   🔄 Processing video frames...")
        final_video = concatenate_videoclips(video_clips)
        
        # Save output next to the script
        output_file = "test_stock_video.mp4"
        
        print(f"🎬 Exporting to: {output_file}")
        final_video.write_videofile(
            output_file, 
            fps=30, 
            codec="libx264", 
            preset="ultrafast" # Keep it fast for testing
        )
        
        # Cleanup
        for clip in video_clips: clip.close()
        final_video.close()
        print("\n✨ SUCCESS! Video saved as test_stock_video.mp4")
        
    except Exception as e:
        print(f"❌ Video editing failed: {e}")

if __name__ == "__main__":
    print("\n")
    print("Provide the absolute path to a generated blog markdown file.")
    print("Example: C:/Users/SHAHKAR/Multi_Agent_Blog_generator_FYP/Agents_backend/blogs/ai_in_healthcare/content/blog.md")
    target_blog = input("> ").strip()
    
    if target_blog:
        # Clean up quotes if dragged-and-dropped in terminal
        target_blog = target_blog.strip('"\'') 
        generate_video_from_blog(target_blog)
