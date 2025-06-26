import os
import json
import yt_dlp
import asyncio
import requests
from collections import OrderedDict
from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse, parse_qs

mcp = FastMCP(
    name="YouTube_Video_QA_System",
    instructions=(
        "You are an assistant that answers questions about YouTube videos. "
        "When a user asks a question about a video, first check if the subtitles are already stored locally. "
        "If not, extract and store the subtitles automatically. "
        "Use the subtitles to understand the video content and provide accurate answers. "
        "All responses should be based strictly on the information available in the subtitles."
    )
)

# Path to store subtitles
save_path = os.path.join(os.getcwd(), "dumped_data", "saved_subtitles.json")

# ------------------------- Helper Functions ---------------------------- #

def save_data(save_path, video_id, subs_data, keep=5):
    """Save up to `keep` most recent subtitles in a JSON file."""
    try:
        with open(save_path, "r+", encoding="utf-8") as f:
            try:
                dict_data = json.load(f, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError:
                dict_data = OrderedDict()

            if len(dict_data) >= keep:
                dict_data.pop(next(iter(dict_data)))  # Remove oldest

            dict_data[video_id] = subs_data
            f.seek(0)
            f.truncate()
            json.dump(dict_data, f, indent=4)
    except FileNotFoundError:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(OrderedDict({video_id: subs_data}), f, indent=4)

def get_youtube_video_id(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get("v", [None])[0]

def fetch_timedtext_subtitles(timedtext_url: str) -> str:
    """Fetch subtitles from YouTube's timedtext URL and return with timestamps."""
    try:
        response = requests.get(timedtext_url)
        response.raise_for_status()
        data = response.json()
        subtitles = ""
        for event in data.get("events", []):
            if "segs" in event and "tStartMs" in event:
                time = int(event["tStartMs"]) / 1000
                text = "".join(seg.get("utf8", "") for seg in event["segs"])
                if text.strip():
                    subtitles += f"{time} | {text.strip()}\n"
        return subtitles
    except Exception as e:
        return f"Error fetching subtitles: {e}"

# ------------------------- Main Tools ---------------------------- #

# @mcp.tool(
#     name="Extract_YouTube_Subtitles",
#     description="Downloads subtitles with timestamps from a YouTube video in the specified language."
# )
async def extract_youtube_subtitles(video_url: str, lang: str = "en") -> str:
    """
    Extracts and stores subtitles from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        lang (str): Subtitle language code (default is 'en').

    Returns:
        str: Subtitles with timestamps or an error message.
    """
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "dump_single_json": True,
    }
    try:
        video_id = get_youtube_video_id(video_url)
        if video_id:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subs = info.get("automatic_captions") or info.get("subtitles")

                if not subs or lang not in subs:
                    return "No subtitles found for the requested language."

                # Avoid 'ios' clients
                entries = [entry for entry in subs[lang] if entry.get("__yt_dlp_client") != "ios"]
                if not entries:
                    return "No suitable subtitle source found."

                subtitle_url = entries[0]["url"]
                subtitle_text = fetch_timedtext_subtitles(subtitle_url)

                subtitles = f"Title: {info.get('title', 'Unknown')}\n\nSubtitles:\n{subtitle_text}"
                save_data(save_path, video_id, subtitles)
                return subtitles
        else:
            return "Invalid YouTube video URL."
    except Exception as e:
        return f"Error extracting subtitles: {e}"


@mcp.tool(
    name="Get_YouTube_Subtitles",
    description="Returns subtitles for a YouTube video. Checks locally first; if not found, downloads and stores them with timestamps."
)
async def load_or_extract_youtube_subtitles(video_url: str, lang: str = "en") -> str:
    """
    Retrieves subtitles for a YouTube video to support user Q&A.

    - First checks if subtitles are already stored locally.
    - If not, downloads and stores them in the specified language with timestamps.
    - Returns the subtitle text, based on which user queries can be answered.

    Args:
        video_url (str): The full YouTube video URL.
        lang (str): Language code for subtitles (default is 'en').

    Returns:
        str: Subtitles with timestamps, or an error message.
    """
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return "Invalid YouTube video URL."

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            dict_data = json.load(f, object_pairs_hook=OrderedDict)

        if video_id in dict_data:
            return dict_data[video_id]

        # If not found, extract subtitles
        # subtitles = await mcp.call_tool("Extract_YouTube_Subtitles", {
        #     "video_url": video_url
        # })
        subtitles = await extract_youtube_subtitles(video_url,lang=lang)
        return subtitles

    except FileNotFoundError:
        # subtitles = await mcp.call_tool("Extract_YouTube_Subtitles", {
        #     "video_url": video_url
        # })
        subtitles = await extract_youtube_subtitles(video_url,lang=lang)
        return subtitles
    except Exception as e:
        return f"Unexpected error while checking subtitles: {e}"

# ------------------------- Run MCP ---------------------------- #

if __name__ == "__main__":
    mcp.run(transport="stdio")
