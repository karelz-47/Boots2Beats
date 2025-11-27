import os
import json
from typing import Optional, Dict, Any, List

import streamlit as st
from openai import OpenAI


# === CONFIG ===

# Read the API key from environment variable or Streamlit secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY is not set. "
        "Please configure it in Streamlit Secrets or as an environment variable."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Choose model – you can change later
MODEL_NAME = "gpt-5.1"


# === PROMPT BUILDER ===

def build_prompt(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> str:
    """
    Build the instruction string for the model.
    We ask Boots to Beats (the assistant) to:
    - use web search,
    - find relevant line dance choreographies,
    - and respond as a JSON object only.
    """

    artist_part = f' by "{artist}"' if artist else ""
    region_part = region if region else "any"

    return f"""
You are Boots to Beats, an expert line dance assistant.

You help dancers figure out which line dance choreographies go with specific songs.

USER REQUEST:
- Song: "{song_title}"{artist_part}
- Requested level: {level}
- Requested region: {region_part}
- Max number of choreographies: {max_results}

TASK:
1. Use web search to find ACTUAL LINE DANCE CHOREOGRAPHIES (step sheets, demo/tutorial videos,
   or dance descriptions) that are clearly linked to this specific song.
2. Prefer choreographies that:
   - Explicitly mention the song and/or artist in the title or description, and
   - Match the requested level as closely as possible:
       * Beginner, High Beginner, Improver, Intermediate, Advanced
   - Are suitable or commonly used in the requested region (if inferable).
3. Exclude:
   - General news articles about the song.
   - Non-dance content.
   - Choreographies for completely different songs.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a single JSON object, no extra text, with exactly this structure:

{{
  "song": "{song_title}",
  "artist": "{artist or ""}",
  "requested_level": "{level}",
  "requested_region": "{region or ""}",
  "choreographies": [
    {{
      "rank": 1,
      "name": "Name of the choreography",
      "estimated_level": "Beginner | High Beginner | Improver | Intermediate | Advanced | Unknown",
      "estimated_region": "EU | US | UK | Global | Unknown",
      "type": "step_sheet | tutorial_video | article | other",
      "url": "https://...",
      "extra_sources": [
        "https://... (optional, other relevant links)"
      ],
      "reason": "Short explanation why this is a good match (fit to level/region/song, popularity, etc.)"
    }}
    // Up to {max_results} items, ranked from best to worst
  ]
}}

The JSON must be valid (no trailing commas, no comments) and must not contain any additional fields.
"""


def call_boots_to_beats(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> Dict[str, Any]:
    """
    Single call to OpenAI Responses API with web_search tool.
    Asks for JSON and returns parsed dict.
    """

    prompt = build_prompt(song_title, artist, level, region, max_results)

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
        tools=[{"type": "web_search"}],
        response_format={"type": "json_object"},  # ask the model for JSON
    )

    # Extract the JSON text from the response.
    # The Responses API returns a list in `output`.
    # Each item has `content`; we expect one text output.
    try:
        output_item = response.output[0]
        text_parts = output_item.content
        # text_parts is a list; we expect the first to be text
        text = ""
        if text_parts and hasattr(text_parts[0], "text"):
            text = text_parts[0].text
        else:
            # fallback to string representation
            text = str(response)
    except Exception as e:
        raise RuntimeError(f"Unexpected response structure from OpenAI: {e}")

    # Parse JSON
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{text}") from e

    return data


# === STREAMLIT UI ===

st.set_page_config(page_title="Boots to Beats", page_icon="💃", layout="centered")

st.title("Boots to Beats")
st.write(
    "From track to steps in one search. "
    "Enter a song and I’ll help you find line dance choreographies that fit its beat, "
    "your level, and your dance floor."
)

with st.form("search_form"):
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

    submitted = st.form_submit_button("Find choreographies")

if submitted:
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
            # Re-map to a simple table
            rows = []
            for ch in choreos:
                rows.append({
                    "Rank": ch.get("rank"),
                    "Name": ch.get("name"),
                    "Level": ch.get("estimated_level"),
                    "Region": ch.get("estimated_region"),
                    "Type": ch.get("type"),
                    "URL": ch.get("url"),
                    "Reason": ch.get("reason"),
                })

            st.dataframe(rows, use_container_width=True)

        with st.expander("Raw JSON response"):
            st.json(data)
