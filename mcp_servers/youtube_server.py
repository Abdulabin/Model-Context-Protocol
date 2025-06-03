import json
import yt_dlp
import requests
from collections import OrderedDict
from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse, parse_qs
import asyncio

mcp = FastMCP("YoutubeSubtitleExtractor")

############ HELPER FUNCTIONS ###########################################
def save_data(save_path, video_id, subs_data, keep=5):
    try:
        with open(save_path, "r+", encoding="utf-8") as f:
            try:
                dict_data = json.load(f, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError:
                dict_data = OrderedDict()

            if len(dict_data) >= keep:
                first_key = next(iter(dict_data))
                dict_data.pop(first_key)

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
    """
    Fetches subtitles from YouTube's timedtext API and returns them with timestamps..
    """
    try:
        response = requests.get(timedtext_url)
        response.raise_for_status()
        data = response.json()
        events = data.get("events", [])
        subtitles = ""
        for event in events:
            if "segs" in event and "tStartMs" in event:
                time = int(event["tStartMs"]) / 1000
                text = "".join(seg.get("utf8", "") for seg in event["segs"])
                if text.strip():
                    subtitles += f"{time} | {text.strip()}\n"
        return subtitles
    except Exception as e:
        return f"Error fetching subtitles: {e}"

###################  MAIN TOOLS ############################################################

# @mcp.tool(
#     name="download_subs",
#     description="Extracts subtitles with timestamps for a YouTube video.",
# )
def get_subtitles(video_url: str, save_path: str="D:\my_projects\Gen_AI_Projects\model_context_protocol\mcp_project_one\dumped_data\saved_subtitles.json", lang: str = "en") -> str:
    """
    Downloads subtitles and their timestamps from a YouTube video.

    Args:
        video_url (str): The URL of the YouTube video.
        save_path (str): Path to store the subtitle JSON data.
        lang (str): Language code of the subtitles (e.g., 'en', 'hi'). Default is 'en'.

    Returns:
        str: Subtitles with timestamps, or an error message if subtitles are unavailable.
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
        vedio_id = get_youtube_video_id(video_url)
        if vedio_id:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                subs = info.get("automatic_captions") or info.get("subtitles")

                if not subs or lang not in subs:
                    return "No subtitles found for the requested language."

                # Filter out 'ios' subtitle sources
                entries = [entry for entry in subs[lang] if entry.get("__yt_dlp_client") != "ios"]
                if not entries:
                    return "No suitable subtitle source found (non-iOS only)."

                subtitle_url = entries[0]["url"]
                subtitle_text = fetch_timedtext_subtitles(subtitle_url)

                subtitles =  f"Title: {info.get('title', 'Unknown')}\n\nSubtitles : {subtitle_text}"
                save_data(save_path, vedio_id, subtitles)
                return subtitles
                # return mcp.get_prompt("summarize_youtube_vedio",{"subtitles":subtitles})
        else:
            return "Invalid YouTube video URL. Please provide a correct link."
    except Exception as e:
        return f"Error extracting subtitles: {e}"

@mcp.tool(
    name="get_youtube_vedio_stored_subtitles",
    description="Checks if subtitles for a given YouTube video are already stored locally. If not found, it auto-fetches them."
)
async def get_stored_subtitles(
    video_url: str,
    save_path: str = r"D:\my_projects\Gen_AI_Projects\model_context_protocol\mcp_project_one\dumped_data\saved_subtitles.json"
) -> str:
    """
    Checks if subtitles for a YouTube video have already been stored.
    If not, it triggers subtitle extraction automatically using another MCP tool.

    Args:
        video_url (str): The full YouTube video URL.
        save_path (str): Path to the local JSON file storing subtitles.

    Returns:
        str: Stored subtitles if available, or fetched subtitles from YouTube.
    """
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        return "Invalid YouTube video URL. Please provide a correct link."

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            dict_data = json.load(f, object_pairs_hook=OrderedDict)

        if video_id in dict_data:
            return dict_data[video_id]

        # 🔁 Subtitles not found locally, call the extraction tool
        subtitles=  await mcp.call_tool("download_subs", {
                                                "video_url": video_url,
                                                "save_path": save_path 
                                            })
        return 
    
    except FileNotFoundError:
        # File doesn't exist yet, trigger subtitle extraction
        subtitles =  await mcp.call_tool("download_subs", {
                            "video_url": video_url,
                            "save_path": save_path  
                        })
        return subtitles
    except Exception as e:
        return f"Unexpected error while checking stored subtitles: {e}"




if __name__ == "__main__":
    mcp.run(transport="stdio")
