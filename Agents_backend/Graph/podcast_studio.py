import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI Client
# Note: Ensure OPENAI_API_KEY is in your .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# ============================================================================
# 1. SCRIPT GENERATION (GPT-4o)
# ============================================================================

def write_podcast_script(topic: str, blog_content: str) -> List[Dict[str, str]]:
    """
    Uses GPT-4o to turn the blog into a dialogue script.
    """
    print(f"   ✍️ Writing Podcast Script for: {topic[:30]}...")

    # Truncate to avoid context limit
    safe_content = blog_content[:15000]

    system_prompt = """You are a podcast producer. Convert the provided Blog Post into a engaging podcast transcript.
    
    CHARACTERS:
    - Host (Voice: Onyx): Deep, energetic, male. Main narrator.
    - Guest (Voice: Nova): Insightful, calm, female. Adds details.

    INSTRUCTIONS:
    - Length: Approx 2 minutes (250-300 words).
    - Style: Conversational, "NPR style", natural.
    - Format: JSON List of objects: [{"speaker": "Host", "text": "..."}, ...]
    - Do not make up facts not in the blog.
    """

    user_prompt = f"TOPIC: {topic}\n\nCONTENT:\n{safe_content}"

    try:
        from pydantic import BaseModel, Field
        
        class ScriptLine(BaseModel):
            speaker: str = Field(description="Host or Guest")
            text: str = Field(description="The spoken text")

        class Script(BaseModel):
            dialogue: List[ScriptLine]

        structured_llm = llm.with_structured_output(Script)
        
        result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        return [{"speaker": line.speaker, "text": line.text} for line in result.dialogue]

    except Exception as e:
        print(f"   ❌ Script Gen Failed: {e}")
        return []

# ============================================================================
# 2. AUDIO GENERATION (OpenAI TTS-1)
# ============================================================================

def generate_audio_segments(script: List[Dict[str, str]], output_dir: Path) -> List[str]:
    """
    Generates individual audio clips for each line of dialogue.
    """
    print("   🗣️ Synthesizing Audio (OpenAI TTS)...")
    
    files = []
    
    for i, line in enumerate(script):
        speaker = line["speaker"]
        text = line["text"]
        
        # Select Voice
        # Onyx = Deep Male, Nova = Soft Female
        voice = "onyx" if "Host" in speaker else "nova"
        
        filename = output_dir / f"seg_{i:03d}.mp3"
        
        try:
            # FIXED: Using standard streaming write to avoid DeprecationWarning
            response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text
            )
            
            # Save to file using binary write
            with open(filename, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
            
            files.append(str(filename))
            
        except Exception as e:
            print(f"   ⚠️ Segment {i} failed: {e}")

    return files

def combine_audio_files(file_paths: List[str], output_path: str):
    """
    Stitches audio files together using PyDub.
    """
    print("   🎚️ Mixing Final Podcast...")
    
    if not file_paths:
        print("   ❌ No audio segments to combine.")
        return

    try:
        from pydub import AudioSegment
        
        combined = AudioSegment.empty()
        silence = AudioSegment.silent(duration=500) # 0.5s pause between speakers
        
        for f in file_paths:
            try:
                segment = AudioSegment.from_mp3(f)
                combined += segment + silence
            except Exception as e:
                print(f"   ⚠️ Skipping bad file {f}: {e}")
            
        combined.export(output_path, format="mp3")
        print(f"   ✅ Podcast Saved: {output_path}")

    except ImportError:
        print("   ⚠️ PyDub/FFMPEG not found. Using simple concatenation.")
        # Fallback: Binary append
        with open(output_path, 'wb') as outfile:
            for fname in file_paths:
                with open(fname, 'rb') as infile:
                    outfile.write(infile.read())
    
    # Cleanup temp files
    for f in file_paths:
        try:
            os.remove(f)
        except:
            pass

# ============================================================================
# 3. NODE ENTRY POINT
# ============================================================================

def podcast_node(state: Dict) -> Dict:
    """LangGraph Node."""
    print("--- 🎙️ PODCAST STATION ---")
    
    topic = state.get("topic", "podcast")
    blog_content = state.get("final", "")
    
    if not blog_content:
        print("   ⚠️ No blog content found.")
        return {"audio_path": None}

    # 1. Script
    script = write_podcast_script(topic, blog_content)
    if not script: return {"audio_path": None}
    
    # 2. Setup Folder
    output_dir = Path("generated_podcasts")
    output_dir.mkdir(exist_ok=True)
    
    # 3. Generate Segments
    segment_files = generate_audio_segments(script, output_dir)
    
    if not segment_files:
        return {"audio_path": None}

    # 4. Combine
    safe_topic = topic.replace(" ", "_").replace(":", "").replace("/", "")[:40]
    final_path = output_dir / f"podcast_{safe_topic}.mp3"
    
    combine_audio_files(segment_files, str(final_path))
    
    return {"audio_path": str(final_path)}

# ============================================================================
# TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    # Test Data
    test_state = {
        "topic": "Test Podcast",
        "final": "This is a test blog post content. AI is amazing."
    }
    podcast_node(test_state)
# TEST RUNNER
# ============================================================================
if __name__ == "__main__":
    # Test Data
    test_state = {
        "topic": "# Best T20 Batter of All Time: Ranking the Legends of T20 Cricket",
        "final": """

T20 cricket has transformed the global sporting landscape, captivating over 1.5 billion fans worldwide and generating unprecedented commercial and cultural impact. Its fast-paced, high-intensity format has redefined how cricket is played and consumed, making the role of the batter more crucial than ever. Yet, identifying the best T20 batter of all time remains a complex challenge. The format’s evolution—marked by changing rules, diverse playing conditions, and a new generation of dynamic players—complicates direct comparisons across eras.

The difficulty lies not only in comparing raw statistics but also in evaluating adaptability, consistency under pressure, and influence on match outcomes. For instance, legends who dominated early T20 World Cups set benchmarks that current stars strive to surpass, while emerging talents continuously reshape the game’s strategic dimensions. According to recent analyses, players like Virat Kohli and Babar Azam have redefined batting excellence, yet newcomers such as Abhishek Sharma have rapidly ascended to the top of ICC rankings, illustrating the sport’s ongoing evolution [Cricket365](https://www.cricket365.com/t20-cricket/most-runs-in-t20-world-cup-history), [ABP Live](https://news.abplive.com/sports/cricket/rise-of-abhishek-sharma-to-world-no-1-t20i-batter-key-achievements-milestones-1825583).

This discussion will explore the historical legends who laid the foundation for T20 batting greatness, analyze the current top performers shaping the 2026 landscape, and clarify the criteria—such as strike rate, average, and match-winning impact—that define true excellence in this exhilarating format. Understanding these dimensions is essential for appreciating the artistry and skill that distinguish the best T20 batters in cricket history.

As the 2026 T20 World Cup approaches, the quest to crown the greatest batter intensifies, inviting a deeper examination of what it truly means to excel in the shortest format of the game [ICC](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/five-must-watch-batters-set-to-shine-at-the-t20-world-cup).

T20 cricket revolutionized the sport by condensing the game into a fast-paced, 20-overs-per-side format that demands aggressive yet strategic batting. Unlike Test or One-Day Internationals, T20 matches last approximately three hours, intensifying pressure on batsmen to score quickly while preserving wickets. This unique challenge requires a blend of power-hitting, innovation, and adaptability to varying match situations.

### Key Batting Metrics in T20 Cricket

Strike rate is the foremost metric defining a T20 batter’s effectiveness. It measures runs scored per 100 balls faced, reflecting the ability to accelerate scoring. A high strike rate often correlates with match-winning performances, as rapid run accumulation can shift momentum decisively. Batting average, which calculates runs per dismissal, remains important but is often secondary to strike rate in this format. Consistency—frequent contributions across matches—adds value by providing reliability under pressure. Impact, a more qualitative measure, assesses how a batter’s innings influences the game’s outcome, such as scoring crucial runs during powerplays or death overs.

### Evolution of T20 Batting and Its Significance

Since its inception, T20 batting has evolved from cautious stroke play to an aggressive, boundary-focused approach. Early players prioritized survival; modern batters employ innovative shots like the scoop, reverse sweep, and switch hit to exploit field placements. This evolution reflects advancements in technique, fitness, and mental resilience, raising the standard for what defines a top T20 batter. Understanding these dynamics is essential when ranking players, as it contextualizes statistics within the era and style of play. For instance, contemporary rankings highlight players who combine high strike rates with consistency and match impact, as seen in the ICC Player Rankings for T20 batsmen [NDTV Sports](https://sports.ndtv.com/cricket/icc-rankings/t20-batting).

The transformation of batting strategies underscores why evaluating T20 legends requires a nuanced approach that balances raw numbers with situational performance. This foundation sets the stage for analyzing the greatest T20 batters and their contributions to the sport’s most thrilling format.

The evolution of T20 batting owes much to a select group of players whose performances redefined the format’s possibilities. Among these historical legends, Virat Kohli stands out for his unparalleled consistency and impact in T20 World Cups. Kohli’s ability to anchor innings while maintaining a high strike rate has made him a cornerstone of India’s batting lineup. He holds the record for the most runs scored in T20 World Cup history, a testament to his sustained excellence across multiple tournaments [Cricket365](https://www.cricket365.com/t20-cricket/most-runs-in-t20-world-cup-history). His technique and temperament under pressure have set a benchmark for modern T20 batsmanship.

### Mahela Jayawardene: Pioneer of T20 Elegance

Mahela Jayawardene’s influence on early T20 cricket is profound, particularly through his defining innings that combined classical stroke play with innovative shot-making. His performances during the inaugural years of the format helped establish a template for stylish yet effective batting in T20s. Jayawardene’s ability to adapt to the fast-paced demands of T20 cricket while maintaining composure was instrumental in Sri Lanka’s competitive edge. His innings often balanced aggression with strategic pacing, showcasing a nuanced understanding of the game’s evolving dynamics.

### Other Notable Legends and Their Contributions

Beyond Kohli and Jayawardene, players like Rohit Sharma have also significantly shaped T20 batting. Sharma’s explosive starts and capacity to accelerate innings have made him one of the most feared openers in the format. His multiple centuries in T20 internationals highlight his ability to dominate bowling attacks consistently. Collectively, these players have contributed to raising the standard of T20 batting, blending power, precision, and adaptability.

These historical figures not only amassed impressive statistics but also influenced the tactical approach to T20 batting, inspiring a generation of players to innovate and excel. Their legacies continue to inform contemporary strategies and player development in the shortest format of the game.

As the T20 format evolves, examining these legends provides critical insight into the foundations of batting excellence that current and future stars build upon.

Abhishek Sharma has emerged as the preeminent T20 batter in 2026, securing the No. 1 spot in the ICC T20 batting rankings through a combination of consistency and explosive scoring. His recent performances have been marked by a remarkable strike rate exceeding 145 and an average hovering around 38 in international T20 matches. Notably, Sharma’s ability to accelerate innings during critical phases has propelled his team to numerous victories, underscoring his value in high-pressure situations. His record includes multiple half-centuries in major tournaments, and he has been instrumental in setting or chasing challenging targets, reflecting a blend of technical skill and strategic acumen [ABP Live](https://news.abplive.com/sports/cricket/rise-of-abhishek-sharma-to-world-no-1-t20i-batter-key-achievements-milestones-1825583).

### Comparing the Elite: Phil Salt, Jos Buttler, and Tilak Varma

Alongside Sharma, Phil Salt and Jos Buttler remain formidable contenders in the top echelons of T20 batting. Salt is renowned for his aggressive left-handed stroke play, boasting a strike rate near 150 and a penchant for boundary hitting that destabilizes bowling attacks early in the innings. His performances in recent T20 World Cups have been pivotal, often providing explosive starts that set the tone for his team’s innings [ICC](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/five-must-watch-batters-set-to-shine-at-the-t20-world-cup).

Jos Buttler, meanwhile, combines power-hitting with tactical versatility. His ability to adapt to different match situations—whether accelerating the run rate or stabilizing the innings—makes him a unique asset. Buttler’s strike rate, typically around 140, is complemented by a high boundary percentage and a strong record in knockout matches, highlighting his clutch performance capability [NDTV Sports](https://sports.ndtv.com/cricket/icc-rankings/t20-batting).

Tilak Varma represents the rising generation of T20 batters, blending classical technique with modern aggression. His strike rate of approximately 135 is balanced by a solid average, reflecting a maturity beyond his years. Varma’s performances in domestic leagues and international fixtures have demonstrated his potential to challenge established stars, particularly through his innovative shot selection and composure under pressure [USA Today](https://www.usatoday.com/story/sports/2026/02/06/t20-cricket-world-cup-2026-15-players-to-watch/88301234007/).

### Distinct Styles and Tournament Impact

While all four players excel in scoring quickly, their playing styles differ significantly. Sharma’s approach is characterized by calculated aggression, often pacing his innings to maximize impact during the death overs. Salt’s game is more explosive from the outset, frequently taking the attack to bowlers in the powerplay. Buttler’s adaptability allows him to switch gears seamlessly, making him effective across various match contexts. Varma’s style is a hybrid, combining patience with sudden bursts of power, which has proven effective in fluctuating match scenarios.

In terms of tournament performances, Sharma’s consistency in the 2026 T20 World Cup has been unmatched, contributing to his rise to the top ranking. Salt and Buttler have also delivered match-winning innings, with Buttler’s experience shining in pressure situations. Varma’s emergence as a dependable middle-order batter adds depth to his team’s lineup, signaling a promising future.

The nuanced differences in strike rates, averages, and situational effectiveness among these batters illustrate the evolving dynamics of T20 cricket, where versatility and adaptability are as crucial as raw power. This competitive landscape sets the stage for an exciting era in T20 batting excellence [Spoda.ai](https://www.spoda.ai/blog/icc-t20-mens-batters-ranking-2026).

As the 2026 season progresses, the performances of these top batters will continue to shape the narrative of T20 cricket’s greatest legends.

Evaluating the best T20 batter of all time requires a multifaceted approach that balances quantitative data with qualitative insights. Statistical measures form the foundation of this assessment, with runs scored serving as a primary indicator of a batter’s productivity. However, sheer volume of runs alone does not capture the full picture. Strike rate, which measures scoring speed, is equally critical in the fast-paced T20 format, reflecting a player’s ability to accelerate the innings. Batting average provides insight into consistency, while performance under pressure—such as in tight chases or knockout matches—highlights mental resilience and clutch ability.

### Statistical Measures and Tournament Impact

Beyond individual statistics, impact in major tournaments like the T20 World Cup is a vital criterion. Success on cricket’s biggest stage often distinguishes great players from legends. For instance, the top run-scorers in T20 World Cup history have demonstrated not only skill but also the capacity to perform when stakes are highest [Cricket365](https://www.cricket365.com/t20-cricket/most-runs-in-t20-world-cup-history). Consistent contributions in these tournaments enhance a batter’s legacy, as they face the strongest opposition and intense pressure.

### Intangibles: Adaptability, Match-Winning Ability, and Longevity

Subjective factors also play a crucial role. Adaptability to different pitch conditions, bowling attacks, and match situations is essential in T20 cricket’s dynamic environment. A batter’s ability to change gears—whether stabilizing an innings or launching an aggressive assault—reflects tactical acumen. Match-winning ability, often measured by decisive innings that turn games, underscores a player’s influence beyond raw numbers. Longevity further cements greatness; sustaining high performance over multiple seasons and evolving with the game demonstrates exceptional skill and dedication [ICC](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/five-must-watch-batters-set-to-shine-at-the-t20-world-cup).

Combining these objective and subjective criteria ensures a comprehensive evaluation of T20 batters, recognizing not only statistical excellence but also the intangible qualities that define cricketing legends. This framework sets the stage for ranking the all-time greats in the fast-evolving world of T20 cricket.

The debate over the best T20 batter of all time hinges on a nuanced evaluation of consistency, impact, adaptability, and record-breaking performances. Historical legends such as Virat Kohli have set a high benchmark with their sustained excellence and match-winning contributions. Kohli’s ability to anchor innings and maintain a remarkable average in T20 internationals underscores his enduring relevance in the format. His consistency is reflected in ICC rankings, where he remains a top contender due to his prolific run-scoring and crucial performances under pressure [ICC Player Rankings for T20 Batsmen 2026](https://sports.ndtv.com/cricket/icc-rankings/t20-batting).

### Comparing Legends and Emerging Stars

While veterans like Kohli and others have dominated the T20 landscape for years, the emergence of new talents has reshaped the conversation. Abhishek Sharma’s meteoric rise exemplifies this shift. Sharma’s record-breaking achievements, including becoming the world No. 1 T20I batter, highlight his exceptional skill and adaptability in the shortest format. His aggressive stroke play combined with a high strike rate has made him a formidable opponent, capable of changing the course of a game rapidly [Rise Of Abhishek Sharma To World No. 1 T20I Batter - ABP Live](https://news.abplive.com/sports/cricket/rise-of-abhishek-sharma-to-world-no-1-t20i-batter-key-achievements-milestones-1825583).

Moreover, Sharma’s performances in recent T20 World Cups have been instrumental in his ascent, showcasing his ability to perform on the biggest stages. This contrasts with some established stars who, while consistent, have not always matched the explosive impact Sharma delivers in crucial moments [Revealed: The 9 best T20 World Cup batters of all time](https://www.cricket365.com/t20-cricket/most-runs-in-t20-world-cup-history).

### Consistency Versus Impact: The Ongoing Contenders

The evaluation of the best T20 batter must balance consistency with the capacity to influence outcomes decisively. Players like Kohli exemplify consistency, often anchoring innings and accumulating runs steadily. In contrast, Sharma and other emerging stars bring a dynamic, high-impact style that can shift momentum instantly. This duality ensures that the title of the best T20 batter remains contested, with both historical legends and current stars offering compelling cases.

Ultimately, the synthesis of these criteria suggests that while Sharma’s rise is remarkable and positions him as a leading candidate, the sustained excellence and match-winning pedigree of players like Kohli maintain their status as top contenders [ICC T20 Men's Batters Ranking 2026 - Spoda.ai](https://www.spoda.ai/blog/icc-t20-mens-batters-ranking-2026).

As the T20 format continues to evolve, the debate over the greatest batter will likely intensify, driven by emerging talents and the enduring brilliance of established legends.

Mastering the appreciation of T20 batting excellence requires more than casual viewing; it demands focused observation and informed analysis. One effective approach is to watch matches with an emphasis on individual batting techniques—paying close attention to shot selection, strike rotation, and adaptability under pressure. Observing how batters handle different bowling styles and pitch conditions reveals the nuances that distinguish great players from the rest.

### Leveraging ICC Rankings and Key Tournaments

Following the International Cricket Council (ICC) player rankings offers a dynamic way to track the performance and consistency of top T20 batters globally. The ICC T20 batting rankings are updated regularly, reflecting players’ recent form and impact in matches, making them an essential resource for fans seeking to identify emerging talents and established legends alike [NDTV Sports](https://sports.ndtv.com/cricket/icc-rankings/t20-batting). Additionally, major tournaments such as the T20 World Cup serve as prime stages where batting prowess is tested against the best competition. Keeping abreast of these events, including player previews and expert analyses, enriches the viewing experience and deepens understanding of the game’s evolving standards [ICC](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/five-must-watch-batters-set-to-shine-at-the-t20-world-cup).

### Deepening Understanding Through Batting Metrics

To truly appreciate batting excellence, fans should familiarize themselves with key performance metrics beyond runs scored. Strike rate, boundary percentage, and consistency under varying match situations provide a more comprehensive picture of a batter’s effectiveness. For example, analyzing how a player’s strike rate fluctuates against spin versus pace bowling can highlight technical strengths or vulnerabilities. Engaging with statistical platforms and expert commentary that break down these metrics allows fans to move beyond surface-level appreciation and develop a nuanced perspective on batting mastery [Spoda.ai](https://www.spoda.ai/blog/icc-t20-mens-batters-ranking-2026).

By combining attentive match viewing, following authoritative rankings, and studying detailed batting statistics, fans can elevate their engagement with T20 cricket and recognize the true legends of the format. How might these insights change the way one watches the next high-stakes T20 encounter?

The quest to identify the best T20 batter of all time involves a nuanced evaluation of multiple factors, including consistency, strike rate, adaptability across conditions, and impact in high-pressure situations. No single metric can capture the full scope of excellence in this fast-paced format. Instead, a comprehensive assessment must consider both statistical achievements and the ability to innovate under evolving game dynamics, as highlighted by the top performers in T20 World Cup history [Cricket365](https://www.cricket365.com/t20-cricket/most-runs-in-t20-world-cup-history).

### The Ever-Changing Landscape of T20 Batting

T20 cricket remains one of the most dynamic formats, continuously shaped by emerging talents and strategic innovations. Players like Abhishek Sharma, who recently ascended to the world No. 1 T20I batter spot, exemplify how new stars can redefine batting paradigms through aggressive yet calculated play [ABP Live](https://news.abplive.com/sports/cricket/rise-of-abhishek-sharma-to-world-no-1-t20i-batter-key-achievements-milestones-1825583). The upcoming T20 World Cup tournaments promise to showcase a fresh wave of batters poised to leave their mark, underscoring the importance of staying informed about player performances and rankings [ICC](https://www.icc-cricket.com/tournaments/mens-t20-world-cup-2026/news/five-must-watch-batters-set-to-shine-at-the-t20-world-cup).

As the format evolves, so too does the definition of greatness. Enthusiasts and analysts alike must remain engaged with the latest developments, appreciating both the legends who have shaped the game and the rising stars who will carry it forward. Following upcoming tournaments and player trajectories will provide invaluable insights into the future of T20 batting excellence [NDTV Sports](https://sports.ndtv.com/cricket/icc-rankings/t20-batting).

The continuous evolution of T20 batting invites us to celebrate past achievements while eagerly anticipating the innovations yet to come. How will the next generation of batters redefine the art of T20 cricket?
"""
    }
    podcast_node(test_state)