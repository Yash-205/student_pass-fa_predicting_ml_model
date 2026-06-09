"""
Web Search Tool for the Study Coach.
Uses the Tavily API to search for online resources, educational videos,
and additional academic help beyond the internal knowledge base.
"""

import os
from typing import List
from tavily import TavilyClient

class WebSearchTool:
    """
    A wrapper around the Tavily search client to fetch live web resources.
    """
    def __init__(self, api_key: str = None):
        """Initializes the Tavily client using the provided or environment API key."""
        self.client = TavilyClient(api_key=api_key or os.getenv("TAVILY_API_KEY"))

    def search(self, query: str, max_results: int = 3) -> List[str]:
        """
        Performs a web search and returns formatted titles and URLs.
        
        Args:
            query: The search query string.
            max_results: Maximum number of search results to return.
            
        Returns:
            List of strings in the format "Title — URL".
        """
        try:
            results = self.client.search(query)
            formatted = []
            for r in results["results"][:max_results]:
                title = r.get("title", "Resource")
                url = r.get("url", "")
                formatted.append(f"{title} — {url}")
            return formatted
        except Exception as e:
            print(f"Web search failed: {e}")
            return []

