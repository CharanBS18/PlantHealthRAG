# Plant Health RAG (Offline-First)

Local Streamlit app for plant disease guidance using:
- Retrieval-Augmented Generation (RAG) over a local knowledge file
- Optional image symptom analysis
- Simple safety guardrails
- Optional text-to-speech output

## What this does
- Accepts farmer symptom text and/or uploaded plant image.
- Retrieves the most relevant disease/nutrient entries from `data/plant_knowledge.txt`.
- Generates simple, action-oriented advice in 3 sections:
  - Disease Name
  - What You Should Do Now
  - Precautions
- Shows top matches with confidence and saves analysis history.
- Supports output translation (English, Hindi, Telugu).

## Tech stack
- Python + Streamlit UI
- FAISS for local vector search
- `sentence-transformers` for embeddings
- Hugging Face `transformers` pipelines for text and vision
- Optional `pyttsx3` / macOS `say` for TTS

## Project structure
```text
.
├── streamlit_app.py               # Streamlit UI + app orchestration
├── config.py                      # Env-based runtime configuration
├── data/
│   ├── plant_knowledge.txt        # Source knowledge base
│   └── analysis_history.json      # Saved diagnosis history
├── embeddings/
│   ├── index.faiss                # Generated vector index
│   └── documents.pkl              # Chunked docs + metadata
├── services/
│   ├── vector_store.py            # Build/load/search vector DB
│   ├── rag_pipeline.py            # Main RAG flow
│   ├── image_analyzer.py          # Vision pipelines and image hints
│   ├── prompt_builder.py          # Prompt template
│   ├── safety_guardrails.py       # Safety note generation
│   ├── language_support.py        # Translation support/fallbacks
│   ├── voice_output.py            # TTS generation
│   └── history_store.py           # History persistence
├── tests/
│   └── test_rules.py              # Unit tests
├── scripts/
│   └── setup_offline.sh           # Local setup helper
└── requirements.txt
```

## Prerequisites
- Python 3.10+ (recommended)
- `pip`
- Enough disk/RAM for local Hugging Face models

## Setup
### Option 1: quick setup script
```bash
bash scripts/setup_offline.sh
```

### Option 2: manual setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration
Edit `.env` as needed:
```env
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL_NAME=google/flan-t5-base
LLM_MAX_NEW_TOKENS=256
LLM_TEMPERATURE=0.2
VISION_MODEL_NAME=Salesforce/blip-image-captioning-base
HISTORY_PATH=data/analysis_history.json
DEFAULT_LANGUAGE=English
ENABLE_TTS=false
```

## Run the app
```bash
streamlit run streamlit_app.py
```

Then in the UI:
1. Click **Build Knowledge Base Index** (required first run or after KB changes).
2. Upload image and/or type symptoms.
3. Click **Analyze Plant**.
4. Optionally click **Generate Audio Advice**.

## Knowledge base format
`data/plant_knowledge.txt` is plain text with disease entries like:
```text
Tomato Early Blight:
Crop: Tomato
Stage: Vegetative to Fruiting
Region: Warm humid regions
Cause: ...
Symptoms: ...
Treatment: ...
Prevention: ...
```

`Crop`, `Stage`, and `Region` fields are parsed as metadata for filtered retrieval.

## Deployment

### Option 1: Streamlit Cloud (Recommended)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repository (`CharanBS18/PlantHealthRAG`)
4. Click Deploy

**Note:** The app uses large ML models that may require a paid plan for sufficient resources.

### Option 2: Heroku
1. Create a Heroku account and app
2. Add the following secrets to your GitHub repository:
   - `HEROKU_API_KEY`: Your Heroku API key
   - `HEROKU_APP_NAME`: Your Heroku app name
   - `HEROKU_EMAIL`: Your Heroku email
3. Push this code to the `main` branch to trigger automatic deployment

**Note:** Free Heroku dynos have limited resources and may not handle the ML model loading well.

## Tests
```bash
pytest -q
```

## Notes on "offline"
- Inference is local once models and dependencies are available.
- First-time model download from Hugging Face may require internet.
- After model caching, usage can be offline.

## Troubleshooting
- `Vector index not found`: click **Build Knowledge Base Index**.
- `transformers` or `sentence-transformers` errors: reinstall with `pip install -r requirements.txt`.
- Slow first run: local model initialization/download is expected.
- No TTS output: set `ENABLE_TTS=true`; macOS fallback uses `say` and `afconvert`.
