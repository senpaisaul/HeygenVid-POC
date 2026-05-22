# 🎬 UGC Video Ad Generator

Turn any product into a scroll-stopping UGC video ad using AI.

## Pipeline

```
User Input ──→ OpenAI (Creative Director) ──→ HeyGen (Video Agent) ──→ Video
```

1. **User** provides a product link, uploads assets (images/videos/PDFs), and writes a brief description of the ad they want
2. **User** picks an avatar and voice from HeyGen's public catalogue
3. **OpenAI GPT-4o** analyzes everything and crafts a detailed generation prompt — deciding scene direction, hook, script, pacing, product showcase moments, and CTA
4. **HeyGen Video Agent** receives the prompt + avatar + voice + assets and generates the full video
5. **User** watches and downloads the finished UGC ad

## Setup

```bash
pip install -r requirements.txt
```

## API Keys Required

| Key | Where to get it |
|-----|----------------|
| HeyGen API Key | [app.heygen.com → Settings → API](https://app.heygen.com/home?from=&nav=API) |
| OpenAI API Key | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |

## Run

```bash
streamlit run app.py
```

Keys can be entered in the sidebar UI, or set as environment variables:

```bash
export HEYGEN_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

## Cost Estimates

- **OpenAI**: ~$0.01-0.03 per generation (GPT-4o, ~1500 tokens)
- **HeyGen**: ~$0.0333/sec of output video (Video Agent)
  - 30s video ≈ $1.00
  - 60s video ≈ $2.00
- **Asset uploads**: Free (included)
- **Avatar creation**: $1.00/call (only if creating custom — public library is free)

## Tech Stack

- **Streamlit** — UI
- **OpenAI GPT-4o** — Creative direction / prompt engineering
- **HeyGen API v3** — Video Agent for generation, Avatars API for catalogue, Voices API for selection, Assets API for uploads
