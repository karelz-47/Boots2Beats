import os
import json
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI


# ============= CONFIG & CLIENT ============= #

# Page config must come before other Streamlit calls
st.set_page_config(
    page_title="Boots to Beats",
    page_icon="logo.png",  # logo.png in the same folder as app.py
    layout="centered",
)

# Read API key from env or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or (
    st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
)

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. Please add it as an environment variable "
        "or in Streamlit Secrets (Settings → Secrets)."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4.1-mini"  # adjust if needed


# ============= SHARED JSON HELPER ============= #

def extract_json_block(s: str) -> str:
    """
    Extract the first top-level JSON object from a text string.
    We assume the model returns exactly one object.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output.")
    return s[start: end + 1]


# ============= PROMPTS ============= #

def build_prompt_dedicated(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> str:
    """
    Prompt for PART 1: dedicated choreographies + song_info.
    """
    artist_part = f' by "{artist}"' if artist else ""
    region_part = region if region else "any"

    return f"""You are Boots to Beats, an expert line dance assistant.

You help dancers figure out which line dance choreographies go with specific songs.

USER REQUEST:
- Song: "{song_title}"{artist_part}
- Requested level: {level}
- Requested region: {region_part}
- Requested number of choreographies in this group: {max_results}

THIS IS PART 1 OF 2. In this part, focus ONLY on line dance choreographies that are
clearly linked to THIS input song, and provide a short dancer-oriented analysis of the song.

PART 1A – SONG ANALYSIS:
1. Use web search to determine:
   - The approximate tempo/BPM of the input song.
   - The time signature (e.g. 4/4, 3/4).
   - The main dance style / rhythm (e.g. country cha-cha, nightclub two-step, swing).
   - A short description of the musical feel (e.g. "relaxed mid-tempo country waltz").
   - Any commonly referenced social/partner/line-dance styles used with this song.
2. Summarise this in a compact, dancer-friendly description.

PART 1B – DEDICATED CHOREOGRAPHIES:
1. Use web search to identify line dances that:
   - Were choreographed specifically for this song, OR
   - Are widely recognised in the line dance community as THE standard line dance
     for this song.
2. For each suitable choreography, collect:
   - Name
   - Estimated level (Beginner, High Beginner, Improver, Intermediate, Advanced)
   - Region of origin or primary use (if inferable)
   - At least one reliable step-sheet or tutorial link

QUANTITY RULE FOR THIS PART:
- If web search indicates that there are at least {max_results} DISTINCT dedicated
  line dances for this song, you MUST return exactly {max_results} choreographies
  in this group.
- Only return fewer than {max_results} when, after reasonable searching, you genuinely
  cannot identify that many distinct dedicated choreographies for this song.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a single JSON object, no extra text.

The JSON object must have:
- "song": string (song title)
- "artist": string (artist name if known)
- "requested_level": string
- "requested_region": string
- "song_info": object
- "choreographies": array

The "song_info" object must contain:
- "title": string
- "artist": string
- "bpm": number or string (approximate BPM)
- "tempo_label": string (e.g. "slow", "mid-tempo", "up-tempo")
- "style": string (e.g. "country cha-cha", "nightclub two-step")
- "time_signature": string (e.g. "4/4", "3/4")
- "dance_feel": string, short phrase for dancers
- "typical_dance_styles": array of strings (e.g. ["line dance", "two-step"])
- "summary": string (2–3 sentences oriented to dancers)
- "sources": array of strings (optional, URLs you used)

Each item in "choreographies" must be an object with:
- "rank": integer (1 = strongest dedicated match)
- "name": string
- "estimated_level": string
- "estimated_region": string
- "type": string (one of "step_sheet", "tutorial_video", "article", "other")
- "fit_type": string, MUST be "dedicated_for_song"
- "url": string (main learning link)
- "extra_sources": array of strings (optional)
- "reason": string (why this is a good dedicated match for the song)

The JSON must be syntactically valid (no trailing commas, no comments)."""


def build_prompt_generic(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
    song_info: Optional[Dict[str, Any]],
) -> str:
    """
    Prompt for PART 2: ONLY choreographies from OTHER songs,
    but musically compatible with the input song.
    """
    artist_part = f' by "{artist}"' if artist else ""
    region_part = region if region else "any"

    # Provide a short inline summary of song_info to guide the model
    song_info_summary = ""
    if song_info:
        bpm = song_info.get("bpm")
        style = song_info.get("style")
        tempo_label = song_info.get("tempo_label")
        summary_text = song_info.get("summary") or ""
        meta_bits = []
        if bpm:
            meta_bits.append(f"≈{bpm} BPM")
        if tempo_label:
            meta_bits.append(str(tempo_label))
        if style:
            meta_bits.append(str(style))
        meta_line = ", ".join(meta_bits)
        song_info_summary = f"Approximate musical profile: {meta_line}. Summary: {summary_text}"

    return f"""You are Boots to Beats, an expert line dance assistant.

This is PART 2 OF 2 for the same user query.

The user asked for line dance choreographies for:
- Song: "{song_title}"{artist_part}
- Requested level: {level}
- Requested region: {region_part}
- Requested number of choreographies in this group: {max_results}

SONG PROFILE (approximate, from previous analysis):
{song_info_summary}

YOUR TASK IN THIS PART:
Focus ONLY on line dance choreographies that were originally created for OTHER songs,
but which are musically compatible with this input song.

1. Use web search to identify popular line dances whose ORIGINAL music has:
   - Tempo/BPM similar to the input song.
   - Compatible rhythm and style (e.g. similar cha-cha / waltz / two-step / swing feel).
2. These dances do NOT have to mention the input song at all.
3. They should be realistic alternate choices for a DJ or instructor to use when
   this input song is playing.
4. For each choreography, clearly describe in "reason":
   - The original song and approximate BPM/style.
   - Why that makes it a good musical match for the input song.

QUANTITY RULE FOR THIS PART:
- If web search indicates that there are at least {max_results} DISTINCT suitable
  alternate choreographies (for other songs), you MUST return exactly {max_results}
  choreographies in this group.
- Only return fewer than {max_results} when, after reasonable searching, you genuinely
  cannot identify that many distinct suitable alternates.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a single JSON object, no extra text.

The JSON object must have:
- "song": string (song title, same input song)
- "artist": string (artist name if known)
- "requested_level": string
- "requested_region": string
- "choreographies": array

Each item in "choreographies" must be an object with:
- "rank": integer (1 = best alternate match)
- "name": string
- "estimated_level": string
- "estimated_region": string
- "type": string (one of "step_sheet", "tutorial_video", "article", "other")
- "fit_type": string, MUST be "compatible_generic"
- "url": string (main learning link)
- "extra_sources": array of strings (optional)
- "reason": string, describing:
    * the original song and style,
    * why it is musically appropriate for the input song.

The JSON must be syntactically valid (no trailing commas, no comments)."""


# ============= OPENAI CALL WRAPPER ============= #

def call_model_with_web_search(prompt: str) -> Dict[str, Any]:
    """
    Call the OpenAI Responses API with web_search tool and parse JSON output if possible.
    If parsing fails (no JSON), return a dict with '_raw_text' containing the full text.
    """
    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    # Try the helper if available
    try:
        text = response.output_text
    except Exception:
        # Fallback: manually collect text parts
        parts: List[str] = []
        try:
            for item in response.output:
                for content in getattr(item, "content", []):
                    if hasattr(content, "text") and content.text is not None:
                        parts.append(content.text)
        except Exception:
            # As a last resort, just dump the raw response
            return {"_raw_text": str(response)}
        text = "\n".join(parts)

    # Try to parse JSON; if it fails, return raw text instead of crashing
    try:
        json_str = extract_json_block(text)
        data = json.loads(json_str)
        return data
    except Exception:
        return {"_raw_text": text}


# ============= RENDER HELPERS ============= #

def render_choreo_group(title: str, dances: List[Dict[str, Any]]) -> None:
    """Render a group of choreographies as mobile-friendly cards."""
    if not dances:
        return

    st.markdown(f"### {title}")

    for ch in dances:
        rank = ch.get("rank", "?")
        name = ch.get("name", "Unknown choreography")
        level_str = ch.get("estimated_level", "Unknown")
        region_str = ch.get("estimated_region", "Unknown")
        type_str = ch.get("type", "other")
        fit_type = ch.get("fit_type", "unknown")
        url = ch.get("url")
        extra_sources = ch.get("extra_sources", []) or []
        reason = ch.get("reason", "")

        with st.container():
            st.markdown(f"**#{rank} – {name}**")

            meta_line = (
                f"Level: {level_str} · Region: {region_str} · "
                f"Type: {type_str} · Fit: {fit_type}"
            )
            st.markdown(meta_line)

            if url:
                st.markdown(f"[Open choreography ↗]({url})")

            if extra_sources:
                first_extra = extra_sources[0]
                st.markdown(f"[Extra source ↗]({first_extra})")

            if reason:
                st.markdown(f"> {reason}")

            st.markdown("---")


def render_song_info(song_info: Dict[str, Any]) -> None:
    """Render a short card about the input song: tempo, style, etc."""
    if not song_info:
        return

    title = song_info.get("title") or "Song info"
    artist = song_info.get("artist") or ""
    bpm = song_info.get("bpm")
    tempo_label = song_info.get("tempo_label")
    style = song_info.get("style")
    time_sig = song_info.get("time_signature")
    dance_feel = song_info.get("dance_feel")
    typical_styles = song_info.get("typical_dance_styles") or []
    summary = song_info.get("summary")
    sources = song_info.get("sources") or []

    st.markdown("### About this song")

    header_line = f"**{title}**"
    if artist:
        header_line += f" – {artist}"
    st.markdown(header_line)

    meta_parts: List[str] = []
    if bpm:
        meta_parts.append(f"≈ {bpm} BPM")
    if tempo_label:
        meta_parts.append(str(tempo_label))
    if style:
        meta_parts.append(str(style))
    if time_sig:
        meta_parts.append(str(time_sig))

    if meta_parts:
        st.markdown(" · ".join(meta_parts))

    if dance_feel:
        st.markdown(str(dance_feel))

    if typical_styles:
        st.markdown(
            "Typical dance styles: " + ", ".join(str(x) for x in typical_styles)
        )

    if summary:
        st.markdown(f"> {summary}")

    if sources:
        first = sources[0]
        st.markdown(f"[Source info ↗]({first})")

    st.markdown("---")


# ============= STREAMLIT UI ============= #

# Logo + title
if os.path.exists("logo.png"):
    st.image("logo.png", width=220)

st.title("Boots to Beats")
st.write(
    "From track to steps in one search. "
    "Enter a song and I’ll help you find line dance choreographies that fit its beat, "
    "your level, and your dance floor."
)

# --- Input widgets ---

song_title = st.text_input("Song title", value="Texas Hold 'Em")
artist = st.text_input("Artist (optional)", value="Beyoncé")

level = st.selectbox(
    "Desired level",
    ["Beginner", "High Beginner", "Improver", "Intermediate", "Advanced", "Any"],
    index=0,
)

region_choice = st.selectbox(
    "Region (hint for the model)",
    ["Global", "EU", "US", "UK", "Other"],
    index=1,
)
region_other = st.text_input("If 'Other', specify region (optional)")

if region_choice == "Other" and region_other.strip():
    region_value: Optional[str] = region_other.strip()
elif region_choice == "Global":
    region_value = None
else:
    region_value = region_choice

max_results = st.slider(
    "Choreographies per group",
    min_value=1,
    max_value=5,
    value=3,
)

run_search = st.button("Find choreographies")

# --- Run search when button clicked ---

if run_search:
    if not song_title.strip():
        st.error("Please enter a song title.")
    else:
        song_clean = song_title.strip()
        artist_clean = artist.strip() or None

        # PART 1 – Dedicated choreographies + song_info
        with st.spinner("Finding choreographies dedicated to this song..."):
            dedicated_data = call_model_with_web_search(
                build_prompt_dedicated(
                    song_title=song_clean,
                    artist=artist_clean,
                    level=level,
                    region=region_value,
                    max_results=max_results,
                )
            )

        # Determine if we got structured JSON or just raw text
        dedicated_raw_text = dedicated_data.get("_raw_text") if isinstance(dedicated_data, dict) else None
        if dedicated_raw_text:
            song_info = {}
            dedicated_choreos: List[Dict[str, Any]] = []
        else:
            song_info = dedicated_data.get("song_info", {}) if isinstance(dedicated_data, dict) else {}
            dedicated_choreos = dedicated_data.get("choreographies", []) if isinstance(dedicated_data, dict) else []

        # PART 2 – Musical matches from other songs
        with st.spinner("Finding musical matches from other songs..."):
            generic_data = call_model_with_web_search(
                build_prompt_generic(
                    song_title=song_clean,
                    artist=artist_clean,
                    level=level,
                    region=region_value,
                    max_results=max_results,
                    song_info=song_info,
                )
            )

        generic_raw_text = generic_data.get("_raw_text") if isinstance(generic_data, dict) else None
        if generic_raw_text:
            generic_choreos: List[Dict[str, Any]] = []
        else:
            generic_choreos = generic_data.get("choreographies", []) if isinstance(generic_data, dict) else []

        # Render song info if we have structured data
        if song_info:
            render_song_info(song_info)

        st.subheader("Top matches")

        # Enforce caps from slider for structured lists
        if dedicated_choreos:
            dedicated_choreos = dedicated_choreos[:max_results]
        if generic_choreos:
            generic_choreos = generic_choreos[:max_results]

        # Render structured results as cards
        if dedicated_choreos:
            render_choreo_group("Dances choreographed for this song", dedicated_choreos)
        if generic_choreos:
            render_choreo_group("Musical matches from other songs", generic_choreos)

        # If a call returned only raw text, show it as-is
        if dedicated_raw_text:
            st.markdown("### Dances choreographed for this song (raw model answer)")
            st.markdown(dedicated_raw_text)

        if generic_raw_text:
            st.markdown("### Musical matches from other songs (raw model answer)")
            st.markdown(generic_raw_text)

        if (
            not dedicated_choreos
            and not generic_choreos
            and not dedicated_raw_text
            and not generic_raw_text
        ):
            st.info("No suitable choreographies found (or the model could not identify any).")

        # Raw output for debugging in expanders
        with st.expander("Model output – dedicated group (raw)"):
            if dedicated_raw_text:
                st.text(dedicated_raw_text)
            else:
                st.json(dedicated_data)

        with st.expander("Model output – musical matches group (raw)"):
            if generic_raw_text:
                st.text(generic_raw_text)
            else:
                st.json(generic_data)
