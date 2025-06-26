import os
import json
import arxiv
from typing import List
from mcp.server.fastmcp import FastMCP

# -------------------- Initialize MCP Server --------------------
mcp = FastMCP(
    name="Arxiv_Research_Assistant",
    instructions=(
        "You are an AI assistant that helps users explore research papers from arXiv. "
        "When a user provides a topic, search for the most relevant papers and store the metadata locally. "
        "When a user provides a paper ID, retrieve the stored metadata for that specific paper. "
        "Ensure all responses are based on accurate and relevant metadata from arXiv."
    )
)

save_path = os.path.join(os.getcwd(), "dumped_data", "papers_info.json")

# -------------------- Tool 1: Search Papers --------------------
@mcp.tool(
    name="Search_Arxiv_Papers",
    description="Searches arXiv for relevant research papers on a topic and stores their metadata locally."
)
def search_arxiv_papers(topic: str, max_results: int = 5) -> List[str]:
    """
    Searches arXiv for research papers related to a topic and stores the results.

    Args:
        topic (str): The topic to search for.
        max_results (int): Number of papers to retrieve (default: 5).

    Returns:
        List[str]: List of arXiv paper IDs retrieved.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    try:
        with open(save_path, "r", encoding="utf-8") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in client.results(search):
        paper_id = paper.get_short_id()
        paper_ids.append(paper_id)
        papers_info[paper_id] = {
            "title": paper.title,
            "authors": [author.name for author in paper.authors],
            "summary": paper.summary,
            "pdf_url": paper.pdf_url,
            "published": str(paper.published.date())
        }

    with open(save_path, "w", encoding="utf-8") as json_file:
        json.dump(papers_info, json_file, indent=2)

    return paper_ids

# -------------------- Tool 2: Get Paper Info --------------------
@mcp.tool(
    name="Get_Paper_Info_By_ID",
    description="Retrieves detailed metadata for a specific research paper using its arXiv ID."
)
def get_research_paper_info(paper_id: str) -> str:
    """
    Retrieves detailed metadata for a paper from stored arXiv search results.

    Args:
        paper_id (str): The arXiv ID of the paper.

    Returns:
        str: JSON-formatted metadata or an error message if not found.
    """
    try:
        with open(save_path, "r", encoding="utf-8") as json_file:
            papers_info = json.load(json_file)
        if paper_id in papers_info:
            return json.dumps(papers_info[paper_id], indent=2)
        else:
            return f"No information found for paper ID: {paper_id}."
    except (FileNotFoundError, json.JSONDecodeError):
        return "No papers have been stored yet."

# -------------------- Start MCP Server --------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
