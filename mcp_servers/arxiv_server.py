import os
import json
import arxiv
from typing import List
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("Arxiv_researcher")

@mcp.tool(
    name="search_arxiv_papers",
    description="Searches for recent and relevant research papers from arXiv based on a given topic and stores key metadata locally."
)
def search_arxiv_papers(topic: str, max_results: int = 5, save_path="D:/my_projects/Gen_AI_Projects/model_context_protocol/mcp_project_one/dumped_data/papers_info.json") -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.

    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        save_path: Path to store the paper metadata

    Returns:
        List of paper IDs found in the search
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # path = os.path.join(save_path, topic.lower().replace(" ", "_"))
    # os.makedirs(path, exist_ok=True)

    try:
        with open(save_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    with open(save_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    return paper_ids


@mcp.tool(
    name="get_research_paper_info",
    description="Fetches detailed metadata for a specific paper by ID from locally stored arXiv search results."
)
def get_research_paper_info(paper_id: str, save_path: str = "D:/my_projects/Gen_AI_Projects/model_context_protocol/mcp_project_one/dumped_data/papers_info.json") -> str:
    """
    Search for information about a specific paper across all topic directories.

    Args:
        paper_id: The ID of the paper to look for
        save_path: Path to the JSON file containing saved paper metadata

    Returns:
        JSON string with paper information if found, error message if not found
    """
    if os.path.isfile(save_path):
        try:
            with open(save_path, "r") as json_file:
                papers_info = json.load(json_file)
                if paper_id in papers_info:
                    return json.dumps(papers_info[paper_id], indent=2)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    return f"There's no saved information related to paper {paper_id}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
