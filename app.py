"""
UGC Video Ad Pipeline
User Input → Azure OpenAI Creative Director → HeyGen Video Generation

Flow:
1. User provides product link, uploads assets, writes a brief description
2. User picks an avatar + voice from HeyGen's public library
3. Azure OpenAI (gpt-5-mini / gpt-5-nano) acts as a creative director —
   analyzes everything and writes a rich generation prompt
   (scene direction, tone, pacing, CTA)
4. That prompt + avatar + voice + assets are sent to HeyGen Video Agent
5. We poll until the video is ready and display it
"""

import streamlit as st
import requests
import time
import os
from openai import OpenAI

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="UGC Ad Generator", page_icon="🎬", layout="wide")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 1100px; margin: 0 auto; }
    .stApp { background-color: #0a0a0a; }
    h1 { color: #ffffff; font-weight: 800; }
    h2, h3 { color: #e0e0e0; }
    .avatar-card {
        border: 2px solid #222;
        border-radius: 12px;
        padding: 8px;
        text-align: center;
        cursor: pointer;
        transition: border-color 0.2s;
        background: #111;
    }
    .avatar-card:hover { border-color: #4a9eff; }
    .avatar-card.selected { border-color: #4a9eff; background: #1a2a3a; }
    .status-box {
        background: #111;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    div[data-testid="stFileUploader"] { background: #111; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Azure OpenAI config (env-driven, no UI input for keys) ───────────────────
AZURE_ENDPOINT = os.getenv(
    "AZURE_EASTUS2_ENDPOINT",
    "https://scalistro-test-v1-resource.openai.azure.com/openai/v1",
)
AZURE_API_KEY = os.getenv("AZURE_EASTUS2_API_KEY", "")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2025-04-01-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

# ─── Sidebar: HeyGen key + Azure status + model toggle ────────────────────────
with st.sidebar:
    st.markdown("## 🔑 API Configuration")
    heygen_key = st.text_input(
        "HeyGen API Key", type="password",
        value=os.getenv("HEYGEN_API_KEY", ""),
        help="Get yours at app.heygen.com → Settings → API",
    )

    selected_model = st.radio(
        "Creative director model",
        ["gpt-5-mini", "gpt-5-nano"],
        index=0 if AZURE_DEPLOYMENT == "gpt-5-mini" else 1,
        horizontal=True,
        help="`mini` = better creative quality • `nano` = faster & cheaper",
        captions=["Better quality", "Faster & cheaper"],
    )

    st.divider()
    st.markdown("### How it works")
    st.markdown("""
    1. **You** provide product link, assets & a brief  
    2. **Pick** an avatar & voice from HeyGen  
    3. **Azure OpenAI** crafts a creative ad prompt  
    4. **HeyGen** generates the video  
    5. **Download** your UGC ad 🎉
    """)

HEYGEN_BASE = "https://api.heygen.com"

# ─── Azure OpenAI client ──────────────────────────────────────────────────────
def get_azure_client():
    """
    Azure OpenAI v1 API surface — used via the standard OpenAI SDK by pointing
    base_url at the Azure /openai/v1 endpoint and pinning the api-version.
    """
    if not AZURE_API_KEY:
        raise RuntimeError("AZURE_EASTUS2_API_KEY is not set")
    return OpenAI(
        base_url=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        default_query={"api-version": AZURE_API_VERSION},
    )

# ─── HeyGen helpers ───────────────────────────────────────────────────────────

def heygen_headers():
    return {"X-Api-Key": heygen_key, "Content-Type": "application/json"}


def heygen_get(path, params=None):
    """GET request to HeyGen API."""
    r = requests.get(f"{HEYGEN_BASE}{path}",
                     headers=heygen_headers(), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner="Loading avatars from HeyGen…")
def fetch_avatar_looks(_key, ownership="public", limit=50):
    """Fetch public avatar looks. Cached for 10 min."""
    all_looks = []
    token = None
    pages = 0
    while pages < 3:  # max 3 pages = 150 avatars
        params = {"ownership": ownership, "limit": limit}
        if token:
            params["token"] = token
        resp = requests.get(f"{HEYGEN_BASE}/v3/avatars/looks",
                            headers={"X-Api-Key": _key}, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        all_looks.extend(body.get("data", []))
        if body.get("has_more") and body.get("next_token"):
            token = body["next_token"]
            pages += 1
        else:
            break
    return all_looks


@st.cache_data(ttl=600, show_spinner="Loading voices from HeyGen…")
def fetch_voices(_key, limit=100):
    """Fetch public voices. Cached for 10 min."""
    all_voices = []
    token = None
    pages = 0
    while pages < 3:
        params = {"type": "public", "limit": limit}
        if token:
            params["token"] = token
        resp = requests.get(f"{HEYGEN_BASE}/v3/voices",
                            headers={"X-Api-Key": _key}, params=params, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        all_voices.extend(body.get("data", []))
        if body.get("has_more") and body.get("next_token"):
            token = body["next_token"]
            pages += 1
        else:
            break
    return all_voices


def upload_asset_to_heygen(file_bytes, filename):
    """Upload a file to HeyGen's asset store, return asset_id."""
    r = requests.post(
        f"{HEYGEN_BASE}/v3/assets",
        headers={"X-Api-Key": heygen_key},
        files={"file": (filename, file_bytes)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]["asset_id"]


def generate_creative_prompt(product_link, description, asset_names, model):
    """
    The Azure OpenAI creative director layer.
    Takes the raw user inputs and produces a rich, scene-directed generation
    prompt for HeyGen's Video Agent.
    """
    client = get_azure_client()

    system_msg = """You are an elite UGC video ad creative director. Your job is to take a 
product link, uploaded asset descriptions, and a brief user description, then produce a 
DETAILED video generation prompt for an AI video platform (HeyGen).

Your output prompt must include:
1. **Hook** (first 3 seconds) — what grabs attention immediately
2. **Scene direction** — specific visual actions the avatar should convey 
   (e.g. "avatar holds up the product excitedly", "avatar demonstrates pouring into the pan")
3. **Script** — the exact words the avatar should say, written in a natural UGC tone 
   (casual, authentic, like a real person sharing a recommendation, NOT corporate)
4. **Pacing & tone** — energy level, mood transitions
5. **Product showcase moments** — when/how to feature the product assets
6. **Call-to-action** — compelling closing that drives action
7. **Orientation** — recommend portrait (9:16) for TikTok/Reels or landscape (16:9) for YouTube

Think about what would actually make someone STOP scrolling. Be specific about visual 
storytelling — if it's a food product, describe sizzling, steam, stacking ingredients. 
If it's tech, describe the satisfying UI interactions. If it's fashion, describe the 
confidence transformation.

Output ONLY the generation prompt text (no markdown headers, no explanations). 
This text will be sent directly to HeyGen's Video Agent API as the 'prompt' field.
Keep it under 2000 characters."""

    user_msg = f"""Product link: {product_link}

User's description of the ad they want:
{description}

Uploaded assets: {', '.join(asset_names) if asset_names else 'None provided'}

Generate the video ad prompt."""

    # GPT-5 family on Azure: use `max_completion_tokens`, omit `temperature`
    # (reasoning models default to 1 and may reject custom values).
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_completion_tokens=1500,
    )
    return response.choices[0].message.content.strip()


def create_heygen_video(prompt, avatar_id, voice_id, files_payload, orientation):
    """Send the generation prompt to HeyGen Video Agent."""
    body = {"prompt": prompt}
    if avatar_id:
        body["avatar_id"] = avatar_id
    if voice_id:
        body["voice_id"] = voice_id
    if orientation:
        body["orientation"] = orientation
    if files_payload:
        body["files"] = files_payload

    r = requests.post(
        f"{HEYGEN_BASE}/v3/video-agents",
        headers=heygen_headers(),
        json=body,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"]


def poll_session(session_id):
    """Poll session until video_id is assigned."""
    for _ in range(60):
        r = requests.get(
            f"{HEYGEN_BASE}/v3/video-agents/{session_id}",
            headers=heygen_headers(), timeout=30,
        )
        r.raise_for_status()
        data = r.json()["data"]
        if data.get("video_id"):
            return data["video_id"], data.get("status")
        if data.get("status") == "failed":
            return None, "failed"
        time.sleep(5)
    return None, "timeout"


def poll_video(video_id):
    """Poll video until completed or failed."""
    for _ in range(120):
        r = requests.get(
            f"{HEYGEN_BASE}/v3/videos/{video_id}",
            headers=heygen_headers(), timeout=30,
        )
        r.raise_for_status()
        data = r.json()["data"]
        status = data.get("status")
        if status == "completed":
            return data
        if status == "failed":
            return data
        time.sleep(5)
    return {"status": "timeout"}


# ─── Main UI ───────────────────────────────────────────────────────────────────

st.markdown("# 🎬 UGC Video Ad Generator")
st.markdown("Turn any product into a scroll-stopping UGC video ad in minutes.")
st.divider()

# Check config
if not heygen_key:
    st.warning("⬅️  Enter your HeyGen API key in the sidebar to get started.")
    st.stop()
if not AZURE_API_KEY:
    st.error("Azure OpenAI is not configured. Set the AZURE_* variables in your `.env` and restart.")
    st.stop()

# ─── Step 1: Product Info ─────────────────────────────────────────────────────
st.markdown("## 📦 Step 1 — Your Product")

col1, col2 = st.columns(2)
with col1:
    product_link = st.text_input(
        "Product URL",
        placeholder="https://example.com/my-awesome-product",
        help="Link to your product page, Amazon listing, Shopify store, etc."
    )
with col2:
    orientation_choice = st.selectbox(
        "Video orientation",
        ["portrait", "landscape"],
        index=0,
        help="Portrait (9:16) for TikTok/Reels, Landscape (16:9) for YouTube"
    )

uploaded_files = st.file_uploader(
    "Upload product assets (images, videos, PDFs)",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg", "mp4", "webm", "mp3", "wav", "pdf"],
    help="Product photos, demo clips, packaging shots — anything that shows off your product"
)

ad_description = st.text_area(
    "Describe the ad you want",
    placeholder="E.g.: I want a fun, energetic ad showing someone cooking with my non-stick pan. "
                "Show how nothing sticks to it. Target audience is home cooks aged 25-40. "
                "Keep it casual and authentic like a real TikTok review.",
    height=120,
)

st.divider()

# ─── Step 2: Avatar & Voice Selection ─────────────────────────────────────────
st.markdown("## 🎭 Step 2 — Choose Your Avatar & Voice")

try:
    avatars = fetch_avatar_looks(heygen_key)
    voices = fetch_voices(heygen_key)
except Exception as e:
    st.error(f"Failed to load HeyGen catalogue: {e}")
    st.stop()

# ── Avatar picker ──
st.markdown("### Avatar")
st.caption(f"Showing {len(avatars)} public avatars. Pick one to be your UGC creator.")

# Gender filter
avatar_genders = sorted(set(a.get("gender", "unknown") or "unknown" for a in avatars))
selected_gender = st.selectbox("Filter by gender", ["all"] + avatar_genders)

filtered_avatars = avatars
if selected_gender != "all":
    filtered_avatars = [a for a in avatars if (a.get("gender") or "unknown") == selected_gender]

# Display avatar grid
if "selected_avatar_id" not in st.session_state:
    st.session_state.selected_avatar_id = None
    st.session_state.selected_avatar_name = None

# Paginate avatars — show 12 at a time
AVATARS_PER_PAGE = 12
avatar_page = st.number_input("Avatar page", min_value=1,
                               max_value=max(1, len(filtered_avatars) // AVATARS_PER_PAGE + 1),
                               value=1, label_visibility="collapsed")
start = (avatar_page - 1) * AVATARS_PER_PAGE
page_avatars = filtered_avatars[start:start + AVATARS_PER_PAGE]

cols = st.columns(4)
for i, avatar in enumerate(page_avatars):
    with cols[i % 4]:
        img_url = avatar.get("preview_image_url", "")
        name = avatar.get("name", "Avatar")
        aid = avatar.get("id", "")
        is_selected = st.session_state.selected_avatar_id == aid

        if img_url:
            st.image(img_url, width=140)
        st.caption(name)
        if st.button(
            "✅ Selected" if is_selected else "Select",
            key=f"av_{aid}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
        ):
            st.session_state.selected_avatar_id = aid
            st.session_state.selected_avatar_name = name
            st.rerun()

if st.session_state.selected_avatar_id:
    st.success(f"Avatar: **{st.session_state.selected_avatar_name}**")

st.markdown("---")

# ── Voice picker ──
st.markdown("### Voice")

vcol1, vcol2 = st.columns(2)
with vcol1:
    voice_lang = st.selectbox(
        "Language",
        ["all"] + sorted(set(v.get("language", "") for v in voices if v.get("language"))),
    )
with vcol2:
    voice_gender = st.selectbox(
        "Voice gender",
        ["all"] + sorted(set(v.get("gender", "") for v in voices if v.get("gender"))),
        key="voice_gender_filter",
    )

filtered_voices = voices
if voice_lang != "all":
    filtered_voices = [v for v in filtered_voices if v.get("language") == voice_lang]
if voice_gender != "all":
    filtered_voices = [v for v in filtered_voices if v.get("gender") == voice_gender]

# Build options for selectbox
voice_options = {f"{v['name']} ({v.get('language','?')}, {v.get('gender','?')})": v["voice_id"]
                 for v in filtered_voices if v.get("voice_id")}

selected_voice_label = st.selectbox(
    "Pick a voice",
    ["-- Select --"] + list(voice_options.keys()),
    help="Choose the voice your avatar will use"
)

selected_voice_id = None
if selected_voice_label != "-- Select --":
    selected_voice_id = voice_options[selected_voice_label]

    # Show preview if available
    matching_voice = next((v for v in filtered_voices
                           if v["voice_id"] == selected_voice_id), None)
    if matching_voice and matching_voice.get("preview_audio_url"):
        st.audio(matching_voice["preview_audio_url"], format="audio/mp3")

st.divider()

# ─── Step 3: Generate ─────────────────────────────────────────────────────────
st.markdown("## 🚀 Step 3 — Generate Your Ad")

ready = (
    product_link
    and ad_description
    and st.session_state.selected_avatar_id
    and selected_voice_id
)

if not ready:
    st.info("Fill in the product details, pick an avatar & voice, then hit Generate.")

if st.button("🎬 Generate UGC Video Ad", type="primary",
             use_container_width=True, disabled=not ready):

    # ── Upload assets to HeyGen ──
    files_payload = []
    asset_names = []

    if uploaded_files:
        with st.status("Uploading assets to HeyGen…", expanded=True) as upload_status:
            for uf in uploaded_files:
                st.write(f"Uploading **{uf.name}**…")
                try:
                    asset_id = upload_asset_to_heygen(uf.getvalue(), uf.name)
                    files_payload.append({"type": "asset_id", "asset_id": asset_id})
                    asset_names.append(uf.name)
                    st.write(f"✅ {uf.name} uploaded")
                except Exception as e:
                    st.warning(f"⚠️ Failed to upload {uf.name}: {e}")
            upload_status.update(label="Assets uploaded!", state="complete")

    # Also pass the product link as a URL file for HeyGen context
    if product_link.startswith("http"):
        files_payload.append({"type": "url", "url": product_link})

    # ── Azure OpenAI creative director ──
    with st.status(f"🧠 {selected_model} is crafting your ad concept…", expanded=True) as ai_status:
        try:
            creative_prompt = generate_creative_prompt(
                product_link, ad_description, asset_names, model=selected_model
            )
            st.markdown("**Generated creative prompt:**")
            st.code(creative_prompt, language=None)
            ai_status.update(label="Creative direction ready!", state="complete")
        except Exception as e:
            st.error(f"Azure OpenAI error: {e}")
            st.stop()

    # ── Send to HeyGen ──
    with st.status("📹 Sending to HeyGen Video Agent…", expanded=True) as gen_status:
        try:
            result = create_heygen_video(
                prompt=creative_prompt,
                avatar_id=st.session_state.selected_avatar_id,
                voice_id=selected_voice_id,
                files_payload=files_payload if files_payload else None,
                orientation=orientation_choice,
            )
            session_id = result.get("session_id")
            st.write(f"Session created: `{session_id}`")
            gen_status.update(label="Video generation started!", state="complete")
        except Exception as e:
            st.error(f"HeyGen error: {e}")
            st.stop()

    # ── Poll for video ──
    with st.status("⏳ Waiting for video to render… (this can take 2-5 min)",
                   expanded=True) as poll_status:

        # First poll session for video_id
        st.write("Waiting for video ID…")
        video_id, sess_status = poll_session(session_id)

        if not video_id:
            st.error(f"Session ended with status: {sess_status}")
            st.stop()

        st.write(f"Video ID: `{video_id}` — now rendering…")

        # Poll video for completion
        video_data = poll_video(video_id)
        final_status = video_data.get("status")

        if final_status == "completed":
            poll_status.update(label="🎉 Video ready!", state="complete")
        elif final_status == "failed":
            poll_status.update(label="❌ Generation failed", state="error")
            st.error(f"Failure: {video_data.get('failure_message', 'Unknown error')}")
            st.stop()
        else:
            poll_status.update(label="⏱️ Timed out", state="error")
            st.warning("Video is still rendering. Check your HeyGen dashboard.")
            st.stop()

    # ── Show result ──
    st.divider()
    st.markdown("## 🎉 Your UGC Video Ad")

    video_url = video_data.get("video_url")
    if video_url:
        st.video(video_url)
        st.markdown(f"[⬇️ Download video]({video_url})")

    thumb_url = video_data.get("thumbnail_url")
    if thumb_url:
        st.image(thumb_url, caption="Thumbnail", width=300)

    duration = video_data.get("duration")
    if duration:
        st.metric("Duration", f"{duration:.1f}s")

    page_url = video_data.get("video_page_url")
    if page_url:
        st.markdown(f"[Open in HeyGen →]({page_url})")