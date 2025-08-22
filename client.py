import argparse
import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple

from fastmcp import Client

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # We'll validate at runtime


# -----------------------------
# Gemini helpers
# -----------------------------

def _load_gemini() -> "genai.GenerativeModel":
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. Add it to dependencies and reinstall."
        )
    api_key = "AIzaSyDwAdmLAvvmMqdqa3p-Jp1vW-GFxu6g3FQ"
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Set it to your Gemini API key."
        )
    # Prefer environment API key; require it
    env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not env_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Set it to your Gemini API key."
        )
    api_key = env_key
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    # Provide a system instruction that teaches tool planning
    return genai.GenerativeModel(
        model_name,
        system_instruction=(
            "You are a tool-using chatbot that decides when to call MCP tools.\n"
            "TOOLS AND ARG SCHEMAS (JSON):\n"
            "- get_weather: {\"location\": string, \"unit\": 'metric'|'imperial' (default 'metric')}\n"
            "- serp_search: {\"query\": string, \"num_results\": integer (1..10, default 5)}\n"
            "- wiki_search: {\"query\": string, \"language\": string (default 'en'), \"limit\": integer (1..20, default 5)}\n"
            "- wiki_summary: {\"title\": string, \"language\": string (default 'en')}\n\n"
            "First, output a compact JSON plan matching exactly this schema: \n"
            "{\n  \"action\": \"respond\" | \"call_tool\",\n  \"tool\": null | \"get_weather\" | \"serp_search\" | \"wiki_search\" | \"wiki_summary\",\n  \"args\": object | null,\n  \"final_answer\": string | null\n}\n"
            "Routing guidance: For definitions or encyclopedia-style facts use wiki_search/wiki_summary.\n"
            "For weather queries use get_weather. For news, recent updates, or web pages use serp_search.\n"
            "Rules: \n- If a tool is needed, set action=call_tool and pick exactly one tool and args as per schemas.\n"
            "- If no tool is needed, set action=respond and provide final_answer.\n"
            "- Output ONLY JSON with no extra commentary."
        ),
    )


async def gemini_plan(model: Any, user_input: str) -> Dict[str, Any]:
    # Blocking call -> run in a thread
    def _run() -> str:
        resp = model.generate_content(user_input)
        return resp.text or "{}"

    text = await asyncio.to_thread(_run)
    raw = (text or "").strip()
    # Strip common code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("Plan is not a JSON object")
        return plan
    except Exception:
        # Fall back to simple response
        return {"action": "respond", "tool": None, "args": None, "final_answer": raw}


async def gemini_answer(model: Any, user_input: str, tool_name: str, tool_args: Dict[str, Any], tool_result: Any) -> str:
    def _safe_dumps(o: Any) -> str:
        try:
            return json.dumps(o, ensure_ascii=False, default=str)
        except Exception:
            try:
                return json.dumps(str(o), ensure_ascii=False)
            except Exception:
                return str(o)

    prompt = (
        "User asked: "
        + _safe_dumps(user_input)
        + "\nTool used: "
        + _safe_dumps(tool_name)
        + "\nArguments: "
        + _safe_dumps(tool_args)
        + "\nTool result (JSON):\n"
        + _safe_dumps(tool_result)
        + "\n\nCompose a concise, helpful answer for the user using the tool result."
    )

    def _run() -> str:
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()

    return await asyncio.to_thread(_run)


def _to_jsonable(o: Any) -> Any:
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, list):
        return [_to_jsonable(i) for i in o]
    if isinstance(o, tuple):
        return [_to_jsonable(i) for i in o]
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    # Handle Pydantic RootModel or similar wrappers
    for attr in ("root", "__root__"):
        if hasattr(o, attr):
            try:
                return _to_jsonable(getattr(o, attr))
            except Exception:
                pass
    # pydantic v2
    if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
        dumped = o.model_dump()
        # Unwrap root key if present
        if isinstance(dumped, dict) and len(dumped) == 1 and ("root" in dumped or "__root__" in dumped):
            dumped = dumped.get("root") or dumped.get("__root__")
        return _to_jsonable(dumped)
    # pydantic v1 / dataclass-like
    if hasattr(o, "dict") and callable(getattr(o, "dict")):
        try:
            dumped = o.dict()
            if isinstance(dumped, dict) and len(dumped) == 1 and ("root" in dumped or "__root__" in dumped):
                dumped = dumped.get("root") or dumped.get("__root__")
            return _to_jsonable(dumped)
        except Exception:
            pass
    if hasattr(o, "__dict__"):
        return _to_jsonable(vars(o))
    return str(o)


def _extract_tool_payload(result: Any) -> Any:
    # FastMCP ToolResult may have .data or .text
    if hasattr(result, "data") and result.data is not None:
        return _to_jsonable(result.data)
    if hasattr(result, "text") and result.text is not None:
        return result.text
    # Fallback to string
    return str(result)


# -----------------------------
# Existing utility commands
# -----------------------------

async def list_all(client: Client) -> None:
    tools = await client.list_tools()
    resources = await client.list_resources()
    prompts = await client.list_prompts()
    print("Tools:")
    for t in tools:
        print(f"- {t.name}: {t.description}")
    print("\nResources:")
    for r in resources:
        print(f"- {r.uri}")
    print("\nPrompts:")
    for p in prompts:
        print(f"- {p.name}")


async def call_tool(client: Client, name: str, args: Dict[str, Any]) -> Any:
    result = await client.call_tool(name, args)
    payload = _extract_tool_payload(result)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


# -----------------------------
# Chat REPL with Gemini
# -----------------------------

async def chat_repl(client: Client) -> None:
    model = _load_gemini()

    print("Chatbot (Gemini). You can still use slash commands:")
    print("/tools, /weather, /search, /wiki, /summary, /quit")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in {"/quit", "/exit"}:
            break

        # Keep legacy slash commands
        if line == "/tools":
            await list_all(client)
            continue
        if line.startswith("/weather"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                location = parts[1]
                unit = parts[2] if len(parts) == 3 else "metric"
                await call_tool(client, "get_weather", {"location": location, "unit": unit})
            else:
                print("Usage: /weather <location> [metric|imperial]")
            continue
        if line.startswith("/search"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                query = parts[1]
                num = int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else 5
                await call_tool(client, "serp_search", {"query": query, "num_results": num})
            else:
                print("Usage: /search <query> [num]")
            continue
        if line.startswith("/wiki"):
            parts = line.split(maxsplit=2)
            if len(parts) >= 2:
                query = parts[1]
                limit = int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else 5
                await call_tool(client, "wiki_search", {"query": query, "limit": limit})
            else:
                print("Usage: /wiki <query> [limit]")
            continue
        if line.startswith("/summary"):
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                await call_tool(client, "wiki_summary", {"title": parts[1]})
            else:
                print("Usage: /summary <title>")
            continue

        # Chat path: ask Gemini to plan
        try:
            plan = await gemini_plan(model, line)
        except Exception as e:
            print(f"[Gemini error] {e}")
            continue

        action = (plan.get("action") or "respond").lower()
        tool = plan.get("tool")
        args = plan.get("args") or {}
        if action == "call_tool" and isinstance(tool, str):
            try:
                payload = await call_tool(client, tool, args)
            except Exception as e:
                print(f"[Tool error] {e}")
                continue
            # If tool already returned a final answer string, use it directly
            if isinstance(payload, str):
                print(payload)
            else:
                # Finalize with Gemini
                try:
                    answer = await gemini_answer(model, line, tool, args, payload)
                except Exception as e:
                    print(f"[Gemini error] {e}")
                    continue
                print(answer)
        else:
            # Pure LLM response
            final = plan.get("final_answer")
            if not final:
                # If planner didn't include final_answer, ask Gemini directly
                try:
                    final = await gemini_answer(model, line, "none", {}, {})
                except Exception as e:
                    print(f"[Gemini error] {e}")
                    continue
            print(final)


async def main() -> None:
    parser = argparse.ArgumentParser(description="FastMCP Client (Gemini Chatbot)")
    parser.add_argument("server", nargs="?", default="server.py", help="Server source: path to .py or URL")
    args = parser.parse_args()

    client = Client(args.server)

    async with client:
        await client.ping()
        await chat_repl(client)


if __name__ == "__main__":
    asyncio.run(main())
