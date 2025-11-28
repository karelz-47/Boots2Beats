import os
import json
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI


# ============= CONFIG & CLIENT ============= #

# Page config must come before any other Streamlit calls
st.set_page_config(
    page_title="Boots to Beats",
    page_icon="logo.png",  # uses logo.png in the same folder as app.py
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
MODEL_NAME = "gpt-5.1"  # you can change later


# ============= PROMPT BUILDER ============= #

def build_prompt(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> str:
    """
    Build the instruction string for the model.
    We also ask explicitly for diverse choreographies and no duplicates.
    """

    artist_part = f' by "{artist}"' if artist else ""
    region_part = region if region else "any"

    return (
        "You are Boots to Beats, an expert line dance assistant.\n\n"
        "You help dancers figure out which line dance choreographies go with specific songs.\n\n"
        f"USER REQUEST:\n"
        f'- Song: "{song_title}"{artist_part}\n'
        f"- Requested level: {level}\n"
        f"- Requested region: {region_part}\n"
        f"- Max number of choreographies: {max_results}\n\n"
        "TASK:\n"
        "1. Use web search to find ACTUAL LINE DANCE CHOREOGRAPHIES (step sheets, demo/tutorial videos,\n"
        "   or dance descriptions) that are clearly linked to this specific song.\n"
        "2. Prefer choreographies that:\n"
        "   - Explicitly mention the song and/or artist in the title or description, and\n"
        "   - Match the requested level as closely as possible:\n"
        "       Beginner, High Beginner, Improver, Intermediate, Advanced, or Any.\n"
        "   - Are suitable or commonly used in the requested region (if inferable).\n"
        "3. Aim for DIVERSITY:\n"
        "   - If there are multiple different choreographies for this song, prefer showing\n"
        "     different dances (different choreographers / step patterns).\n"
        "   - Do NOT return several entries for the same choreography just because it has\n"
        "     multiple videos or step-sheet sites.\n"
        "4. Exclude:\n"
        "   - General news articles about the song.\n"
        "   - Non-dance content.\n"
        "   - Choreographies for completely different songs.\n\n"
        "OUTPUT FORMAT (IMPORTANT):\n"
        "Return ONLY a single JSON object, no extra text. The top-level JSON object must have the keys:\n"
        '  - \"song\" (string)\n'
        '  - \"artist\" (string)\n'
        '  - \"requested_level\" (string)\n'
        '  - \"requested_region\" (string)\n'
        '  - \"choreographies\" (array of objects)\n\n"
        'Each item in \"choreographies\" must be an object with the keys:\n'
        '  - \"rank\" (integer, starting at 1 for best match)\n'
        '  - \"name\" (string, name of the choreography)\n'
        '  - \"estimated_level\" (string)\n'
        '  - \"estimated_region\" (string)\n'
        '  - \"type\" (string, exactly one of: \"step_sheet\", \"tutorial_video\", \"article\", \"other\")\n'
        '  - \"url\" (string, main link for learning that choreography)\n'
        '  - \"extra_sources\" (array of strings, optional, other helpful links)\n'
        '  - \"reason\" (string, short explanation why this is a good match)\n\n'
        f'The array \"choreographies\" must contain at most {max_results} items, ordered from best to worst.\n'
        "DIVERSITY & DEDUPLICATION RULES:\n"
        "- Never include two items that are essentially the same choreography.\n"
        "  Treat dances as the same if they have the same name and choreographer, or\n"
        "  clearly describe the same steps, even if there are different videos/sites.\n"
        "- If you find multiple URLs for the same choreography, create ONE item in\n"
        "  \"choreographies\" for that dance. Put the best/most informative link in \"url\"\n"
        "  and put all other useful links into \"extra_sources\".\n"
        "- If you can only find fewer DISTINCT choreographies than the requested maximum,\n"
        "  just return the smaller number (do NOT pad the list with duplicates).\n"
        "The JSON must be valid (no trailing commas, no comments).\n"
    )


# ============= OPENAI CALL (WITH WEB SEARCH) ============= #

def call_boots_to_beats(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> Dict[str, Any]:
    """
    Single call to OpenAI Responses API with web_search tool.
    We CANNOT use JSON mode with web_search, so we:
    - ask for JSON in the prompt,
    - let the model respond in normal text mode,
    - extract the JSON substring manually and parse it.
    """

    prompt = build_prompt(song_title, artist, level, region, max_results)

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=[{"type": "web_search"}],
    )

    # --- Get model text output ---

    try:
        # Newer SDKs may provide this helper
        text = response.output_text
    except Exception:
        # Fallback: manually collect all text pieces
        parts: List[str] = []
        try:
            for item in response.output:
                for content in getattr(item, "content", []):
                    if hasattr(content, "text") and content.text is not None:
                        parts.append(content.text)
        except Exception as e:
            raise RuntimeError(f"Unexpected response structure from OpenAI: {e}")
        text = "\n".join(parts)

    # --- Extract JSON substring ---

    def extract_json_block(s: str) -> str:
        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in model output:\n{s}")
        return s[start: end + 1]

    json_str = extract_json_block(text)

    # --- Parse JSON ---

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw extracted block:\n{json_str}"
        ) from e

    return data


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

# --- Input widgets (no form, so you can search multiple times) ---

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
    "Max choreographies to return",
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
        with st.spinner("Boots to Beats is searching the web and ranking choreographies..."):
            try:
                data = call_boots_to_beats(
                    song_title=song_title.strip(),
                    artist=artist.strip() or None,
                    level=level,
                    region=region_value,
                    max_results=max_results,
                )
            except Exception as e:
                st.error(f"Error while calling OpenAI / parsing response: {e}")
                st.stop()

        st.subheader("Top matches")

        choreos: List[Dict[str, Any]] = data.get("choreographies", [])

        if not choreos:
            st.info("No suitable choreographies found (or the model could not identify any).")
        else:
            # --- Card-style layout, mobile-friendly ---
            for ch in choreos:
                rank = ch.get("rank", "?")
                name = ch.get("name", "Unknown choreography")
                level_str = ch.get("estimated_level", "Unknown")
                region_str = ch.get("estimated_region", "Unknown")
                type_str = ch.get("type", "other")
                url = ch.get("url")
                extra_sources = ch.get("extra_sources", []) or []
                reason = ch.get("reason", "")

                with st.container():
                    st.markdown(f"**#{rank} – {name}**")

                    meta_line = f"Level: {level_str} · Region: {region_str} · Type: {type_str}"
                    st.markdown(meta_line)

                    if url:
                        st.markdown(f"[Open choreography ↗]({url})")

                    if extra_sources:
                        first_extra = extra_sources[0]
                        st.markdown(f"[Extra source ↗]({first_extra})")

                    if reason:
                        st.markdown(f"> {reason}")

                    st.markdown("---")

        # Raw JSON for debugging
        with st.expander("Raw JSON response"):
            st.json(data)
