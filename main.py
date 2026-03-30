"""
SCENEFORGE — Cinematic Story Generator Backend v7.0
Fixed: scene extraction, image generation, better prompts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import requests
import base64
import re
from dotenv import load_dotenv

load_dotenv()

# ── API KEY ─────────────────────────────────────────────────────────────────
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
if not HF_API_KEY:
    print("⚠  WARNING: HUGGINGFACE_API_KEY not set in .env")

# ── ENDPOINTS ────────────────────────────────────────────────────────────────
HF_TEXT_URL  = "https://router.huggingface.co/v1/chat/completions"
HF_IMAGE_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

TEXT_HEADERS  = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}
IMAGE_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="SCENEFORGE API", version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST MODELS ───────────────────────────────────────────────────────────
class StoryRequest(BaseModel):
    characters: List[str]
    genre: str = "Fantasy"
    tone: str = "Epic"
    subgenres: Optional[str] = ""
    story_length: str = "Medium"
    dialogue_intensity: int = 50
    description_depth: str = "Moderate"
    emotional_intensity: int = 60
    pov: str = "Third Person Limited"
    voice: str = "Lyrical"
    structure: str = "Three-Act"
    conflicts: Optional[str] = ""
    plot_devices: Optional[str] = ""
    world_settings: Optional[List[str]] = []
    time_period: str = "Timeless"
    atmospheres: Optional[str] = ""
    literary_devices: Optional[str] = ""
    ending: str = "Bittersweet"
    prose_style: str = "Descriptive"
    archetypes: Optional[str] = ""
    custom_instructions: Optional[str] = ""

class ImageRequest(BaseModel):
    scenes: List[str]
    genre: str = "Fantasy"
    tone: str = "Epic"
    atmospheres: Optional[str] = ""


# ── HELPERS ──────────────────────────────────────────────────────────────────
def build_story_prompt(req: StoryRequest) -> str:
    chars     = ", ".join(req.characters)
    length_map = {"Short": "600–900 words", "Medium": "1000–1400 words", "Long": "1800–2400 words"}
    word_count = length_map.get(req.story_length, "1000–1400 words")
    settings   = ", ".join(req.world_settings) if req.world_settings else "Unspecified"

    return f"""You are a master literary author writing a {req.genre} story.

=== STORY PARAMETERS ===
Characters    : {chars}
Genre         : {req.genre}{(' / ' + req.subgenres) if req.subgenres else ''}
Tone          : {req.tone}
POV           : {req.pov}
Voice         : {req.voice}
Structure     : {req.structure}
Conflict      : {req.conflicts or 'Character vs Character'}
Plot Devices  : {req.plot_devices or 'None specified'}
World Setting : {settings}
Time Period   : {req.time_period}
Atmosphere    : {req.atmospheres or 'Dramatic'}
Literary Dev  : {req.literary_devices or 'None specified'}
Ending        : {req.ending}
Prose Style   : {req.prose_style}
Archetypes    : {req.archetypes or 'None specified'}
Length        : {word_count}
Dialogue      : ~{req.dialogue_intensity}% of story
Description   : {req.description_depth}
Emotion Level : {req.emotional_intensity}%
{('Special Notes: ' + req.custom_instructions) if req.custom_instructions else ''}

=== OUTPUT FORMAT (follow EXACTLY) ===

TITLE: [Your story title here]
SUBTITLE: [A short poetic tagline]

CHARACTER ROLES:
- [Name]: [Role] — [One sentence description]
(repeat for each character)

---

ACT I — [Evocative name]

[Story body here. Use these markers:]
- Scene headings: INT. LOCATION — TIME  or  EXT. LOCATION — TIME
- Dialogue: CHARACTER NAME: dialogue text
- Internal thoughts: *thought text in asterisks*
- Scene transitions: ~~~

ACT II — [Evocative name]

[Continue story...]

ACT III — [Evocative name]

[Resolution...]

---

SCENE_DESCRIPTIONS:
SCENE_1: [A rich visual description (2–3 sentences) of the most cinematic moment from Act I. Describe lighting, mood, characters' appearance, environment. Make it painterly and specific enough to generate an image from.]
SCENE_2: [Most cinematic moment from Act II — equally vivid and detailed]
SCENE_3: [Climax or turning point from Act III]
SCENE_4: [Final image or epilogue — the emotional resolution made visual]

Write the complete story now. Make it immersive, emotionally resonant, and beautifully written."""


def query_text(prompt: str) -> str:
    payload = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.82,
        "max_tokens": 2000,
        "top_p": 0.92,
    }
    res = requests.post(HF_TEXT_URL, headers=TEXT_HEADERS, json=payload, timeout=120)
    if res.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Text model error: {res.text[:300]}")
    data = res.json()
    return data["choices"][0]["message"]["content"]


def extract_title(text: str) -> str:
    m = re.search(r'^TITLE:\s*(.+)', text, re.MULTILINE)
    return m.group(1).strip() if m else "Untitled Story"


def extract_subtitle(text: str) -> str:
    m = re.search(r'^SUBTITLE:\s*(.+)', text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_scenes(text: str) -> List[str]:
    """Parse SCENE_1..SCENE_4 from the dedicated section at end of story."""
    scenes = []
    # Try dedicated section
    block_match = re.search(r'SCENE_DESCRIPTIONS:(.*?)$', text, re.DOTALL | re.IGNORECASE)
    if block_match:
        block = block_match.group(1)
        matches = re.findall(r'SCENE_\d+:\s*(.+?)(?=SCENE_\d+:|$)', block, re.DOTALL)
        for m in matches:
            cleaned = m.strip().replace('\n', ' ')
            if cleaned:
                scenes.append(cleaned[:400])  # cap length

    # Fallback: look for inline SCENE_N: lines
    if not scenes:
        for line in text.split('\n'):
            m = re.match(r'SCENE_\d+:\s*(.+)', line.strip())
            if m:
                scenes.append(m.group(1).strip()[:400])

    # Last fallback: grab prose paragraph chunks
    if not scenes:
        paras = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 80]
        scenes = paras[:4]

    return scenes[:4]  # max 4 scenes


def build_image_prompt(scene: str, genre: str, tone: str, atmospheres: str) -> str:
    atm = atmospheres or "dramatic cinematic lighting"
    return (
        f"{scene} "
        f"Genre: {genre}. Mood: {tone}. Atmosphere: {atm}. "
        "Cinematic concept art, painterly digital illustration, "
        "highly detailed, dramatic composition, 8k resolution, no text."
    )[:500]  # FLUX works best with concise prompts


def query_image(prompt: str):
    try:
        res = requests.post(
            HF_IMAGE_URL,
            headers=IMAGE_HEADERS,
            json={"inputs": prompt, "parameters": {"num_inference_steps": 4}},
            timeout=120
        )
        if res.status_code != 200:
            return None, f"HTTP {res.status_code}: {res.text[:200]}"

        # Detect content type
        content_type = res.headers.get("content-type", "image/jpeg")
        if "png" in content_type:
            mime = "image/png"
        elif "webp" in content_type:
            mime = "image/webp"
        else:
            mime = "image/jpeg"

        img_b64 = base64.b64encode(res.content).decode("utf-8")
        data_url = f"data:{mime};base64,{img_b64}"
        return data_url, None

    except requests.Timeout:
        return None, "Image generation timed out (120s)"
    except Exception as e:
        return None, str(e)


# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "running", "version": "7.0.0"}


@app.post("/generate-story")
def generate_story(req: StoryRequest):
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY not configured.")
    if len(req.characters) < 2:
        raise HTTPException(status_code=400, detail="At least 2 characters required.")

    prompt = build_story_prompt(req)
    story  = query_text(prompt)
    scenes = extract_scenes(story)

    return {
        "title":    extract_title(story),
        "subtitle": extract_subtitle(story),
        "story":    story,
        "scenes":   scenes,          # ← explicit list, no frontend parsing needed
    }


@app.post("/generate-images")
def generate_images(req: ImageRequest):
    if not HF_API_KEY:
        raise HTTPException(status_code=500, detail="HUGGINGFACE_API_KEY not configured.")
    if not req.scenes:
        raise HTTPException(status_code=400, detail="No scenes provided.")

    results = []
    for scene in req.scenes:
        img_prompt = build_image_prompt(scene, req.genre, req.tone, req.atmospheres or "")
        url, error = query_image(img_prompt)
        results.append({"scene": scene, "url": url, "error": error})

    return {"images": results}


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)