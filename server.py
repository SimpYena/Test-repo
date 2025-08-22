import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from fastmcp import FastMCP

from wiki_client import WikipediaClient

from serpapi import SerpApiClient as SerpApiSearch

from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)


load_dotenv()
API_KEY = os.getenv("SERPAPI_API_KEY")


mcp = FastMCP(
    name="Simple-MCP-Server",
    instructions=
    """
    Tools available:
    - get_weather(location, unit): current weather via Open-Meteo (no API key)
    - serp_search(query, num_results): Google search via SerpAPI (SERPAPI_API_KEY required)
    - wiki_search(query, language, limit): search Wikipedia titles
    - wiki_summary(title, language): fetch Wikipedia page summary
    """
)
wikipedia_client = WikipediaClient(language="en", country=None, enable_cache=False)

async def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


@mcp.tool(tags={"public", "weather"})
async def get_weather(location: str, unit: str = "metric") -> str:
    """Get current weather for a location and return a concise, final answer string."""
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo = await _http_get_json(geo_url, {"name": location, "count": 1})
    if not geo or not geo.get("results"):
        raise ValueError(f"Location not found: {location}")

    place = geo["results"][0]
    lat = place["latitude"]
    lon = place["longitude"]

    if unit.lower() == "imperial":
        temperature_unit = "fahrenheit"
        wind_speed_unit = "mph"
        temp_suffix = "°F"
        wind_suffix = "mph"
    else:
        temperature_unit = "celsius"
        wind_speed_unit = "kmh"
        temp_suffix = "°C"
        wind_suffix = "km/h"

    weather_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": ["temperature_2m", "wind_speed_10m", "weather_code"],
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
    }
    data = await _http_get_json(weather_url, params)
    current = data.get("current") or {}

    temp = current.get("temperature_2m")
    wind = current.get("wind_speed_10m")

    name_parts = [p for p in [place.get("name"), place.get("admin1"), place.get("country")] if p]
    nice_name = ", ".join(name_parts) if name_parts else location

    if temp is None or wind is None:
        return f"Current weather for {nice_name} is unavailable right now. Please try again later."

    return f"The current weather in {nice_name} is {temp}{temp_suffix} with a wind speed of {wind} {wind_suffix}."


@mcp.tool(tags={"public", "google_search"})
async def serp_search(
    query: str,
    num_results: int = 10,
    engine: str = "google_light",
    location: Optional[str] = None,
) -> str:
    """Perform a Google search via SerpAPI and return formatted results.

    Args:
        query: The search query text.
        num_results: Number of results to return (mapped to SerpAPI `num`).
        engine: SerpAPI engine to use (default: "google_light").
        location: Optional location bias, e.g. "Austin, TX".

    Returns:
        A formatted string of search results or an error message.
    """

    if not API_KEY:
        return "Error: SERPAPI_API_KEY is not set. Configure it in your environment."

    params: Dict[str, Any] = {
        "api_key": API_KEY,
        "engine": engine,
        "q": query,
        "num": num_results,
    }
    if location:
        params["location"] = location

    try:
        search = SerpApiSearch(params)
        data = search.get_dict()

        # Process organic search results if available
        if "organic_results" in data:
            formatted_results: List[str] = []
            for result in data.get("organic_results", []):
                title = result.get("title", "No title")
                link = result.get("link", "No link")
                snippet = result.get("snippet", "No snippet")
                formatted_results.append(
                    f"Title: {title}\nLink: {link}\nSnippet: {snippet}\n"
                )
            return "\n".join(formatted_results) if formatted_results else "No organic results found"
        else:
            return "No organic results found"

    # Handle HTTP-specific errors
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return "Error: Rate limit exceeded. Please try again later."
        elif e.response.status_code == 401:
            return "Error: Invalid API key. Please check your SERPAPI_API_KEY."
        else:
            return f"Error: {e.response.status_code} - {e.response.text}"
    # Handle other exceptions (e.g., network issues)
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool(tags={"public", "wikipedia"})
def wiki_search(query: str, language: str = "en", limit: int = 10) -> Dict[str, Any]:
    """Search Wikipedia for articles matching a query.
    Accepts optional language code (e.g., 'en', 'de', 'zh-hans').
    """
    logger.info(f"Tool: Searching Wikipedia for: {query} (lang={language})")
    client = (
        wikipedia_client
        if getattr(wikipedia_client, "resolved_language", "en") == language
        else WikipediaClient(language=language, country=None, enable_cache=False)
    )
    results = client.search(query, limit=limit)
    return {
        "query": query,
        "language": language,
        "results": results,
    }


@mcp.tool(tags={"public", "wikipedia"})
async def wiki_summary(title: str, language: str = "en") -> Dict[str, Any]:
    """Get a concise summary for a Wikipedia page title."""
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    data = await _http_get_json(url)
    return {
        "title": data.get("title"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
        "extract": data.get("extract"),
        "thumbnail": data.get("thumbnail", {}).get("source"),
    }


if __name__ == "__main__":
    mcp.run()
