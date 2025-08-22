# Medical MCP: FastMCP 2.0 Server + Client

This repo contains a full MCP setup using FastMCP 2.0:
- An MCP Server exposing tools for weather, web search (SerpAPI), and Wikipedia
- An MCP Client with a simple chat-style REPL that calls those tools

Docs referenced:
- Server: https://gofastmcp.com/servers/server
- Client: https://gofastmcp.com/clients/client

## Requirements
- Python 3.13+
- Dependencies in `pyproject.toml` (`fastmcp`, `httpx`)
- Optional: `SERPAPI_API_KEY` in your environment for web search

## Install

Using pip:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -e .
```

Or with `uv`:

```powershell
uv venv --python 3.13
.\.venv\Scripts\activate
uv pip install -e .
```

## Run the server (STDIO transport)

```powershell
.\.venv\Scripts\activate
python server.py
```

The server defaults to STDIO transport per FastMCP guidance. You typically run it when a client launches it over stdio; you can also run HTTP:

```powershell
python -c "import server; server.mcp.run(transport='http', host='127.0.0.1', port=8000)"
```

## Run the Streamlit UI (recommended)

```powershell
.\.venv\Scripts\activate
streamlit run streamlit_app.py
```

In the left sidebar, paste your `GOOGLE_API_KEY` (for Gemini) and optionally `SERPAPI_API_KEY` (for web search). Then chat in the UI. The app will plan tool calls via Gemini and call your MCP server tools under the hood.

## Run the CLI client (optional)

The CLI client will also start the server via stdio by default (`server.py`).

```powershell
.\.venv\Scripts\activate
python client.py
```

To point the client at a different server source, pass a path or URL (FastMCP infers the transport):

```powershell
python client.py server.py                # local stdio server
python client.py http://localhost:8000/mcp  # HTTP server
```

## Environment variables

- `SERPAPI_API_KEY`: Required for `/search` via SerpAPI. Without it, the tool raises a clear error.
- `GOOGLE_API_KEY` (or `GEMINI_API_KEY`): Required for the Gemini-powered chatbot client.
- Optional: `GEMINI_MODEL` (default `gemini-1.5-flash`).

On PowerShell:

```powershell
$env:SERPAPI_API_KEY = "<your_serpapi_key>"
$env:GOOGLE_API_KEY = "<your_gemini_key>"
$env:GEMINI_MODEL = "gemini-1.5-flash"  # or gemini-1.5-pro
```

## Files
- [server.py](file:///c:/Users/Admin/Documents/medical-mcp/server.py): FastMCP server with tools:
  - `get_weather(location, unit)` — Open-Meteo geocoding + forecast (no API key)
  - `serp_search(query, num_results)` — Google results via SerpAPI
  - `wiki_search(query, language, limit)` — Wikipedia title search
  - `wiki_summary(title, language)` — Wikipedia page summary
- [client.py](file:///c:/Users/Admin/Documents/medical-mcp/client.py): FastMCP client with a Gemini-powered chatbot REPL
- [pyproject.toml](file:///c:/Users/Admin/Documents/medical-mcp/pyproject.toml): Dependencies

## Notes
- Server and client follow the patterns in the FastMCP 2.0 docs for creation and transport inference.
- Weather uses Open-Meteo (free, no key). SerpAPI requires an API key. Wikipedia tools use the REST endpoints.
