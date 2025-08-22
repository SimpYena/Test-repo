import os
import json
import asyncio
from typing import Any, Dict

import streamlit as st

from fastmcp import Client

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None


# -----------------------------
# Gemini helpers
# -----------------------------

def load_gemini():
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. Please install dependencies."
        )
    api_key = (
        st.session_state.get("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY", None)
    )
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY (or GEMINI_API_KEY). Set it in the sidebar."
        )
    genai.configure(api_key=api_key)
    model_name = (
        st.session_state.get("GEMINI_MODEL")
        or os.getenv("GEMINI_MODEL")
        or "gemini-1.5-flash"
    )
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


def strip_fences(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    return raw


def gemini_plan(model: Any, user_input: str) -> Dict[str, Any]:
    resp = model.generate_content(user_input)
    raw = strip_fences(resp.text or "{}")
    try:
        plan = json.loads(raw)
        if not isinstance(plan, dict):
            raise ValueError("Plan is not a JSON object")
        return plan
    except Exception:
        return {"action": "respond", "tool": None, "args": None, "final_answer": raw}


def gemini_answer(model: Any, user_input: str, tool_name: str, tool_args: Dict[str, Any], tool_result: Any) -> str:
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
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()


# -----------------------------
# MCP tool calling (async)
# -----------------------------

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
    if hasattr(o, "model_dump") and callable(getattr(o, "model_dump")):
        dumped = o.model_dump()
        if isinstance(dumped, dict) and len(dumped) == 1 and ("root" in dumped or "__root__" in dumped):
            dumped = dumped.get("root") or dumped.get("__root__")
        return _to_jsonable(dumped)
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


async def mcp_call(server_src: str, tool: str, args: Dict[str, Any]) -> Any:
    client = Client(server_src)
    async with client:
        await client.ping()
        result = await client.call_tool(tool, args)
        if hasattr(result, "data") and result.data is not None:
            return _to_jsonable(result.data)
        if hasattr(result, "text") and result.text is not None:
            return result.text
        return str(result)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Medical MCP Chatbot", page_icon="💬", layout="centered")

with st.sidebar:
    st.header("Settings")
    st.text_input("GOOGLE_API_KEY", type="password", key="GOOGLE_API_KEY")
    st.text_input("GEMINI_MODEL", value=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), key="GEMINI_MODEL")
    st.text_input("SERPAPI_API_KEY", type="password", key="SERPAPI_API_KEY")
    st.caption("Keys are held in session only. For SerpAPI search tool.")

st.title("Medical MCP Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything. I can check weather, search web, or look up Wikipedia.")
if prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare environment for SerpAPI (tool reads env var)
    if st.session_state.get("SERPAPI_API_KEY"):
        os.environ["SERPAPI_API_KEY"] = st.session_state["SERPAPI_API_KEY"]

    try:
        model = load_gemini()
    except Exception as e:
        err = f"Gemini setup error: {e}"
        with st.chat_message("assistant"):
            st.error(err)
        st.session_state.messages.append({"role": "assistant", "content": err})
    else:
        # Planning
        plan = gemini_plan(model, prompt)

        if (plan.get("action") or "respond").lower() == "call_tool" and isinstance(plan.get("tool"), str):
            tool_name = plan.get("tool")
            tool_args = plan.get("args") or {}
            # Execute tool
            try:
                import os as _os
                server_src = _os.path.join(_os.path.dirname(__file__), "server.py")
                tool_result = asyncio.run(mcp_call(server_src, tool_name, tool_args))
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Tool error: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Tool error: {e}"})
            else:
                # If tool returned a final answer string, use it directly; else ask Gemini to compose
                if isinstance(tool_result, str):
                    answer = tool_result
                else:
                    try:
                        answer = gemini_answer(model, prompt, tool_name, tool_args, tool_result)
                    except Exception as e:
                        answer = f"Gemini error when composing answer: {e}"
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Plan & Tool details"):
                        import json as _json
                        st.code(_json.dumps({"plan": plan, "tool_result": tool_result}, ensure_ascii=False, indent=2, default=str), language="json")
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Pure LLM answer
            final = plan.get("final_answer")
            if not final:
                try:
                    final = gemini_answer(model, prompt, "none", {}, {})
                except Exception as e:
                    final = f"Gemini error: {e}"
            with st.chat_message("assistant"):
                st.markdown(final)
                with st.expander("Plan details"):
                    import json as _json
                    st.code(_json.dumps(plan, ensure_ascii=False, indent=2, default=str), language="json")
            st.session_state.messages.append({"role": "assistant", "content": final})
