import os
import json
from typing import Optional, Dict, Any, List, Tuple

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
MODEL_NAME = "gpt-4.1-mini"  # you can change later


# ============= PROMPT BUILDER ============= #

def build_prompt(
    song_title: str,
    artist: Optional[str],
    level: str,
    region: Optional[str],
    max_results: int,
) -> str:
    """
    Build the instruction string for the model,
    with:
    - song_info: short analysis of the song (tempo, style, feel, etc.)
    - two choreography groups:
        * dedicated_for_song  -> choreos for THIS song
        * compatible_generic  -> choreos for OTHER songs but musically compatible
    """

    artist_part = f' by "{artist}"' if artist else ""
    region_part = region if region else "any"
    total_max = max_results * 2

    return f"""You are Boots to Beats, an expert line dance assistant.

You help dancers figure out which line dance choreographies go with specific songs.

USER REQUEST:
- Song: "{song_title}"{artist_part}
- Requested level: {level}
- Requested region: {region_part}
- Max choreographies per group: {max_results}

MAIN GOAL:
1) Give a brief, factual description of the input song that is useful for line dancers
   (tempo/BPM, rhythm, style, feel).
2) Suggest several different line dance choreographies that a DJ or instructor could
   reasonably use when this song is playing.

There are TWO types of suitable choreographies:

1) Song-specific choreographies ("dedicated_for_song"):
   - Dances that were clearly choreographed specifically for this input song
     (title or description strongly links them to this exact track), OR
   - Dances that are widely recognised in line dance communities as THE standard
     line dance for this song.
   - These must have fit_type = "dedicated_for_song".

2) Musically compatible choreographies from other songs ("compatible_generic"):
   - Dances that were originally choreographed for OTHER songs, but whose music
     (tempo/BPM, rhythm, style) is a good match for this input song.
   - Typical examples in principle: a popular cha-cha line dance written for
     "Señorita Margarita" used as a generic cha-cha for other 100–105 BPM
     country/Latin tracks; or a slow country two-step dance written for one ballad
     but usable on other songs with the same feel.
   - These dances do NOT need to mention the input song at all.
   - In the "reason" field you MUST say something like:
       "Originally written for <other song> at <BPM> <style>; suggested here
        because it matches the tempo and feel of '{song_title}'."
   - These must have fit_type = "compatible_generic".

TASK STEP 1 – SONG ANALYSIS:
- Use web search to determine:
  - The approximate tempo/BPM of the input song.
  - The time signature (e.g. 4/4, 3/4).
  - The main dance style / rhythm (e.g. cha-cha, waltz, nightclub, two-step, swing).
  - A short description of the musical feel (e.g. "relaxed mid-tempo country ballad").
  - Any commonly referenced social/partner/line-dance styles used with this song.
- Summarise this in a compact, dancer-friendly description.

TASK STEP 2 – CHOREOGRAPHY SEARCH:
1. Use web search to:
   - Find dedicated_for_song choreographies for this exact song.
   - Find popular line dances for other songs with similar tempo/rhythm/style that
     would reasonably work as alternative dances for this song.
2. Prefer choreographies that:
   - Explicitly mention the song and/or artist in the title or description
     (for dedicated_for_song), OR
   - Are well-known line dances whose original music is very similar in tempo/rhythm
     to the input song (for compatible_generic).
   - Match the requested level as closely as possible: Beginner, High Beginner,
     Improver, Intermediate, Advanced, or Any.
   - Are suitable or commonly used in the requested region (if inferable).
3. Aim for DIVERSITY:
   - Show different dances (different choreographers or noticeably different patterns).
   - Do NOT return several entries for the same choreography just because it has multiple
     videos or step-sheet sites.
4. Exclude:
   - General news articles about the song.
   - Non-dance content.
   - Choreographies for completely different styles (e.g. phrased advanced waltz for
     a simple mid-tempo cha-cha) unless clearly justified.

GROUPING & COUNTS:
- Treat the maximum number of dedicated_for_song choreographies as {max_results}.
- Treat the maximum number of compatible_generic choreographies as {max_results}.
- Your ideal target is to return approximately:
    * {max_results} items with fit_type = "dedicated_for_song", and
    * {max_results} items with fit_type = "compatible_generic".
- If both types exist in reality, you should NOT return 0 for one type.
  It is better to return fewer than {max_results} dedicated_for_song items and include
  some compatible_generic dances than to return only dedicated_for_song dances.
- The combined length of "choreographies" may be up to {total_max}, but never more.
- If you truly cannot identify ANY reasonable compatible_generic dances, you may return
  only dedicated_for_song, but say this explicitly in the "reason" fields.

OUTPUT FORMAT (IMPORTANT):
Return ONLY a single JSON object, no extra text.

The top-level JSON object must have these keys:
- "song" (string)
- "artist" (string)
- "requested_level" (string)
- "requested_region" (string)
- "song_info" (object)
- "choreographies" (array)

The "song_info" object must have these keys:
- "title" (string)
- "artist" (string)
- "bpm" (number or string, approximate tempo in BPM)
- "tempo_label" (string, e.g. "slow", "mid-tempo", "up-tempo")
- "style" (string, e.g. "country cha-cha", "nightclub two-step", "pop waltz")
- "time_signature" (string, e.g. "4/4", "3/4")
- "dance_feel" (string, short phrase describing feel for dancers)
- "typical_dance_styles" (array of strings, e.g. ["line dance", "two-step"])
- "summary" (string, 2–3 sentence summary oriented to dancers)
- "sources" (array of strings, optional, URLs used to infer BPM/style)

Each item in "choreographies" must be an object with these keys:
- "rank" (integer, starting at 1 for best overall match)
- "name" (string, name of the choreography)
- "estimated_level" (string)
- "estimated_region" (string)
- "type" (string, exactly one of: "step_sheet", "tutorial_video", "article", "other")
- "fit_type" (string, exactly one of: "dedicated_for_song", "compatible_generic")
- "url" (string, main link for learning that choreography)
- "extra_sources" (array of strings, optional, other helpful links)
- "reason" (string, short explanation why this is a good match)

DIVERSITY & DEDUPLICATION RULES FOR "choreographies":
- Never include two items that are essentially the same choreography.
  Treat dances as the same if they have the same name and choreographer, or clearly
  describe the same steps, even if they are hosted on different sites/videos.
- If you find multiple URLs for the same choreography, create ONE item in "choreographies":
  put the best/most informative link in "url" and all other useful links into "extra_sources".
- If you can only find fewer DISTINCT choreographies in a group than the requested maximum,
  just return the smaller number and do NOT pad the list with duplicates.

The JSON must be syntactically valid (no trailing commas, no comments)."""


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


# ============= HELPER: RENDER GROUPS AS CARDS ============= #

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
        # Show just the first source as a link to keep it compact
        first = sources[0]
        st.markdown(f"[Source info ↗]({first})")

    st.markdown("---")


# ============= HELPER: SPLIT GROUPS WITH FALLBACK ============= #

def split_choreographies(
    choreos: List[Dict[str, Any]],
    song_title: str,
    max_per_group: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split choreographies into:
    - dedicated_for_song
    - compatible_generic
    - other

    If the model did not provide any compatible_generic items, use a heuristic:
    reclassify some dances that do NOT clearly mention the input song in their
    name/reason into the compatible_generic group, so the second block is useful.
    """

    dedicated: List[Dict[str, Any]] = []
    compatible: List[Dict[str, Any]] = []
    other: List[Dict[str, Any]] = []

    song_lower = (song_title or "").lower()

    # First pass: respect model's fit_type if present
    for c in choreos:
        fit_type = c.get("fit_type")
        if fit_type == "dedicated_for_song":
            dedicated.append(c)
        elif fit_type == "compatible_generic":
            compatible.append(c)
        else:
            other.append(c)

    # Heuristic fallback: if no compatible_generic, try to create some
    if not compatible and song_lower:
        candidates: List[Dict[str, Any]] = []

        # candidates: choreos that don't obviously reference the input song
        for c in dedicated:
            text = (
                (c.get("name") or "") + " " + (c.get("reason") or "")
            ).lower()
            if song_lower not in text:
                candidates.append(c)

        for c in other:
            candidates.append(c)

        # move candidates into compatible until we reach max_per_group or run out
        for c in candidates:
            if len(compatible) >= max_per_group:
                break
            if c in dedicated:
                dedicated.remove(c)
            elif c in other:
                other.remove(c)
            c["fit_type"] = "compatible_generic"
            compatible.append(c)

    # Trim groups to max_per_group if needed
    if len(dedicated) > max_per_group:
        dedicated = dedicated[:max_per_group]
    if len(compatible) > max_per_group:
        compatible = compatible[:max_per_group]

    return dedicated, compatible, other


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
    "Max choreographies per group",
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
                raise

        # Song info section
        song_info = data.get("song_info", {}) if isinstance(data, dict) else {}
        if song_info:
            render_song_info(song_info)

        st.subheader("Top matches")

        choreos: List[Dict[str, Any]] = data.get("choreographies", []) if isinstance(data, dict) else []

        if not choreos:
            st.info("No suitable choreographies found (or the model could not identify any).")
        else:
            dedicated, compatible, other = split_choreographies(
                choreos=choreos,
                song_title=song_title.strip(),
                max_per_group=max_results,
            )

            render_choreo_group("Dances choreographed for this song", dedicated)
            render_choreo_group("Musical matches from other songs", compatible)

            if other:
                render_choreo_group("Other suggestions", other)

        # Raw JSON for debugging
        with st.expander("Raw JSON response"):
            st.json(data)
