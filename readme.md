# 🚀 AI Content Factory

A multi-agent AI pipeline that generates complete, publication-ready blog packages from a single topic prompt — including the blog post, social media content, a podcast audio file, and quality/fact-check reports.

---

## 📐 Architecture

```
Topic Input
    │
    ▼
┌─────────┐    ┌──────────┐    ┌─────────────┐
│  Router │───▶│ Research │───▶│ Orchestrator│  ← HITL interrupt here
└─────────┘    └──────────┘    └─────────────┘
                                      │
                          ┌───────────┼───────────┐
                          ▼           ▼           ▼
                       Worker      Worker      Worker      (parallel)
                          └───────────┼───────────┘
                                      ▼
                                   Reducer
                           (merge → images → final)
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
              Fact Checker      Social Media       Audio Generator
                    └─────────────────┼─────────────────┘
                                      ▼
                                  Evaluator
                                      │
                                      ▼
                              Organized Output
```

**Key design decisions:**
- Built on [LangGraph](https://github.com/langchain-ai/langgraph) for stateful, interruptible agent workflows
- Fan-out parallel section writing using `Send()` + `operator.add` reducer
- Human-in-the-Loop (HITL) interrupt after planning — approve or edit the outline before writing begins
- Structured Pydantic outputs at every agent boundary (no free-text parsing)
- Domain-agnostic prompts — works for tech, health, finance, lifestyle, etc.

---

## 🗂️ Project Structure

```
Agents_backend/
├── main.py                  # CLI entry point + graph builder + file saver
├── App_ui.py                # Streamlit interactive UI
├── API_v1.py                # FastAPI REST API
├── validators.py            # Topic validator + blog quality evaluator
│
├── Graph/
│   ├── state.py             # LangGraph State TypedDict + Pydantic models
│   ├── nodes.py             # All agent node functions
│   ├── templates.py         # System prompts for every agent
│   ├── structured_data.py   # Fact-check report schemas
│   └── podcast_studio.py    # TTS podcast generator
│
└── blogs/                   # Generated output (auto-created)
    └── <topic>_<timestamp>/
        ├── content/         # Main blog markdown
        ├── social_media/    # LinkedIn, YouTube, Facebook posts
        ├── reports/         # Fact-check + quality evaluation
        ├── research/        # Raw evidence JSON
        ├── audio/           # Podcast MP3
        └── metadata/        # Plan JSON + metadata JSON
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- ffmpeg (required for podcast audio stitching)

### 1. Clone & install

```bash
git clone <repo-url>
cd Agents_backend
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in `Agents_backend/`:

```env
OPENAI_API_KEY=sk-...          # Required — powers all LLM agents
TAVILY_API_KEY=tvly-...        # Required — powers web research
GOOGLE_API_KEY=...             # Optional — enables AI image generation
```

### 3. Run

**CLI (with Human-in-the-Loop plan review):**
```bash
cd Agents_backend
python main.py
```

**Streamlit UI:**
```bash
cd Agents_backend
streamlit run App_ui.py
```

**FastAPI server:**
```bash
cd Agents_backend
uvicorn API_v1:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

---

## 🔌 API Usage

### Generate a blog
```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "The Future of Quantum Computing", "auto_approve": true}'
```

**Response:**
```json
{
  "job_id": "3f2e1d...",
  "status": "pending",
  "estimated_time": 120
}
```

### Check status
```bash
curl http://localhost:8000/api/status/3f2e1d...
```

### Download results
```bash
curl -O http://localhost:8000/api/download/3f2e1d...
```

Full API docs auto-generated at `/docs` (Swagger UI).

---

## 🤖 Agents

| Agent | Model | Role |
|-------|-------|------|
| Router | gpt-4.1-mini | Decides research mode and generates search queries |
| Researcher | gpt-4.1-mini | Filters and structures Tavily search results |
| Orchestrator | gpt-4.1-mini | Creates the detailed section-by-section plan |
| Worker (×N) | gpt-4.1-mini | Writes one section in parallel per task |
| Image Planner | gpt-4.1-mini | Decides image placement and generates prompts |
| Image Generator | Gemini 2.5 Flash | Generates and embeds images (optional) |
| Fact Checker | gpt-4.1-mini | Audits claims and scores citation coverage |
| Social Media | gpt-4.1-mini | Produces LinkedIn, YouTube, Facebook content |
| Podcast | gpt-4o-mini + TTS-1 | Writes dialogue script and synthesizes audio |
| Evaluator | gpt-4o-mini | Scores structure, readability, citations, SEO |

---

## 📦 Sample Output

Running on topic `"AI in Healthcare"` produces:

```
blogs/ai_in_healthcare_20260204_174505/
├── content/
│   └── how_ai_is_revolutionizing_healthcare_in_2026.md   (~1,800 words)
├── social_media/
│   ├── linkedin_....txt
│   ├── youtube_....txt
│   └── facebook_....txt
├── reports/
│   ├── fact_check.txt          (Score: 9/10 — READY)
│   └── quality_evaluation.json (Score: 8.3/10)
├── research/
│   └── evidence.json           (8 sources)
└── metadata/
    ├── plan.json
    └── metadata.json
```

---

## ⚠️ Known Limitations

- **Job storage is in-memory** — restarting the API server clears all job history
- **No authentication** on API endpoints — add an API key layer before any public deployment
- **CORS is open** (`allow_origins=["*"]`) — restrict to your frontend origin in production
- **Podcast requires ffmpeg** — gracefully skipped if not installed

---

## 📄 License

Apache 2.0 — see [LICENSE](../LICENSE)