import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote
import requests

import httpx
from fastmcp import FastMCP

from wiki_client import WikipediaClient


logger = logging.getLogger(__name__)



mcp = FastMCP(
    name="MedicalMCP-Server",
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


@mcp.tool(tags={"public", "search"})
async def serp_search(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search the web using SerpAPI. Requires SERPAPI_API_KEY in the environment."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return [{"error": "Missing SERPAPI_API_KEY environment variable"}]

    url = "https://serpapi.com/search.json"
    params = {"engine": "google", "q": query, "num": max(1, min(num_results, 10)), "api_key": api_key}

    try:
        data = await _http_get_json(url, params)
    except Exception as e:
        return [{"error": f"SerpAPI request failed: {str(e)}"}]

    results: List[Dict[str, Any]] = []
    for item in (data.get("organic_results") or [])[: num_results]:
        results.append(
            {
                "title": item.get("title"),
                "link": item.get("link"),
                "snippet": item.get("snippet"),
                "position": item.get("position"),
            }
        )
    if not results:
        return [{"error": "No results found."}]
    return results


@mcp.tool(tags={"public", "wikipedia"})
def wiki_search(query: str, limit: int = 10) -> Dict[str, Any]:
        """Search Wikipedia for articles matching a query."""
        logger.info(f"Tool: Searching Wikipedia for: {query}")
        results = wikipedia_client.search(query, limit=limit)
        return {
            "query": query,
            "results": results
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
