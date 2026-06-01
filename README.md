# 🎬 UGC Video Ad Generator

Turn any product into a scroll-stopping UGC video ad using AI.

## Pipeline

```
User Input ──→ Azure OpenAI (Creative Director) ──→ HeyGen (Video Agent) ──→ Video
```

1. **User** provides a product link, uploads assets (images/videos/PDFs), and writes a brief description of the ad they want
2. **User** picks an avatar and voice from HeyGen's public catalogue
3. **Azure OpenAI (gpt-5-mini)** analyzes everything and crafts a detailed generation prompt — deciding scene direction, hook, script, pacing, product showcase moments, and CTA
4. **HeyGen Video Agent** receives the prompt + avatar + voice + assets and generates the full video
5. **User** watches and downloads the finished UGC ad

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# then fill in your real keys in .env
```

## Configuration

The **HeyGen API key** is entered in the sidebar (or read from `HEYGEN_API_KEY` env).

The **Azure OpenAI config** is read entirely from environment variables — there's no UI input for it:

| Variable | Example |
|---|---|
| `AZURE_API_VERSION` | `2025-04-01-preview` |
| `AZURE_EASTUS2_ENDPOINT` | `https://<resource>.openai.azure.com/openai/v1` |
| `AZURE_EASTUS2_API_KEY` | your Azure OpenAI key |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5-mini` or `gpt-5-nano` |

| Key | Where to get it |
|-----|----------------|
| HeyGen API Key | [app.heygen.com → Settings → API](https://app.heygen.com/home?from=&nav=API) |
| Azure OpenAI | Azure portal → your OpenAI resource → Keys and Endpoint |

## Run

```bash
streamlit run app.py
```

## Cost Estimates

- **Azure OpenAI (gpt-5-mini)**: ~$0.002–$0.01 per generation (much cheaper than gpt-4o)
- **HeyGen**: ~$0.0333/sec of output video (Video Agent)
  - 30s video ≈ $1.00
  - 60s video ≈ $2.00
- **Asset uploads**: Free (included)
- **Avatar creation**: $1.00/call (only if creating custom — public library is free)

## Tech Stack

- **Streamlit** — UI
- **Azure OpenAI (gpt-5-mini)** — Creative direction / prompt engineering
- **HeyGen API v3** — Video Agent for generation, Avatars API for catalogue, Voices API for selection, Assets API for uploads