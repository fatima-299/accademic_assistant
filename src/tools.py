import ast
import json
import math
import operator
import os
import re
import requests
from tavily import TavilyClient
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# To-do list — persistent across restarts via a local JSON file
# ---------------------------------------------------------------------------
TODO_FILE = "todo_list.json"


def _load_todos() -> list:
    if os.path.exists(TODO_FILE):
        try:
            with open(TODO_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_todos(todos: list) -> None:
    try:
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            json.dump(todos, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Calculator — secure AST-based evaluator
# ---------------------------------------------------------------------------
def calculator_tool(expression: str) -> str:
    try:
        expression = expression.strip()

        # Strip natural-language prefixes
        for prefix in ["calculate", "what is", "compute", "evaluate"]:
            if expression.lower().startswith(prefix):
                expression = expression[len(prefix):].strip()

        expression = expression.replace("^", "**")

        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
        }

        allowed_functions = {
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pow": pow,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "factorial": math.factorial,
        }

        allowed_constants = {"pi": math.pi, "e": math.e}

        def eval_node(node):
            if isinstance(node, ast.Expression):
                return eval_node(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return node.value
                raise ValueError("Only numeric constants are allowed.")
            if isinstance(node, ast.Num):          # Python <3.8 compat
                return node.n
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError("Operator not allowed.")
                return allowed_operators[op_type](
                    eval_node(node.left), eval_node(node.right)
                )
            if isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in allowed_operators:
                    raise ValueError("Unary operator not allowed.")
                return allowed_operators[op_type](eval_node(node.operand))
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Invalid function call.")
                func_name = node.func.id
                if func_name not in allowed_functions:
                    raise ValueError(f"Function '{func_name}' is not allowed.")
                args = [eval_node(arg) for arg in node.args]
                return allowed_functions[func_name](*args)
            if isinstance(node, ast.Name):
                if node.id in allowed_constants:
                    return allowed_constants[node.id]
                raise ValueError(f"Name '{node.id}' is not allowed.")
            raise ValueError("Unsupported expression.")

        parsed = ast.parse(expression, mode="eval")
        result = eval_node(parsed)

        # Format: drop unnecessary decimals for whole numbers
        if isinstance(result, float) and result.is_integer():
            return f"Result: {int(result)}"
        return f"Result: {result}"

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ---------------------------------------------------------------------------
# To-do list — persistent, with add / show / delete / clear
# ---------------------------------------------------------------------------
def todo_tool(command: str) -> str:
    todos = _load_todos()
    cmd = command.strip()
    cmd_lower = cmd.lower()

    # ADD
    if cmd_lower.startswith("add task:"):
        task = cmd[len("add task:"):].strip()
        if not task:
            return "Please provide a task after 'add task:'."
        todos.append(task)
        _save_todos(todos)
        return f"Task added: {task}"

    # SHOW / LIST
    if cmd_lower in {"show tasks", "list tasks", "display tasks", "show task", "list task"}:
        if not todos:
            return "Your to-do list is empty."
        lines = [f"{i + 1}. {t}" for i, t in enumerate(todos)]
        return "**Your tasks:**\n" + "\n".join(lines)

    # DELETE  e.g. "delete task: 2"  or  "remove task: 2"
    if cmd_lower.startswith("delete task:") or cmd_lower.startswith("remove task:"):
        prefix = "delete task:" if cmd_lower.startswith("delete task:") else "remove task:"
        idx_str = cmd[len(prefix):].strip()
        try:
            idx = int(idx_str) - 1
            if 0 <= idx < len(todos):
                removed = todos.pop(idx)
                _save_todos(todos)
                return f"Task removed: {removed}"
            return f"No task with number {idx_str}."
        except ValueError:
            return "Please provide a task number, e.g. 'delete task: 2'."

    # CLEAR ALL
    if cmd_lower in {"clear tasks", "clear all tasks", "reset tasks"}:
        _save_todos([])
        return "All tasks cleared."

    return (
        "Available commands:\n"
        "- `add task: <description>` — add a new task\n"
        "- `show tasks` — list all tasks\n"
        "- `delete task: <number>` — remove a task by number\n"
        "- `clear tasks` — remove all tasks"
    )


# ---------------------------------------------------------------------------
# Weather — real Open-Meteo API (no fake fallback)
# ---------------------------------------------------------------------------
def weather_tool(city: str) -> str:
    import re as _re
    try:
        # Strip all common natural-language prefixes with a single regex pass.
        city_clean = _re.sub(
            r"(?i)^\s*(what\s+is\s+the\s+|how\s+is\s+the\s+|"
            r"what(\'s|s)?\s+the\s+|give\s+me\s+the\s+)?"
            r"(weather|forecast|temperature|climat\w*)\s*(in|for|at|of)?\s*(the\s+)?",
            "",
            city.strip(),
        ).strip(" ?.,\'\"")

        # Also handle reversed order: "London weather" or just leftover suffix
        city_clean = _re.sub(
            r"(?i)\s*(weather|forecast|temperature)$", "", city_clean
        ).strip(" ?.,\'\"")

        city_clean = city_clean.strip()

        if not city_clean:
            return "Please provide a city name."

        geo_resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city_clean, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geo_resp.raise_for_status()
        results = geo_resp.json().get("results")
        if not results:
            return f"Could not find location data for '{city_clean.title()}'."

        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc["name"]
        country = loc.get("country", "")

        weather_resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weathercode",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=10,
        )
        weather_resp.raise_for_status()
        data = weather_resp.json()

        cur = data.get("current", {})
        day = data.get("daily", {})

        temp      = cur.get("temperature_2m", "N/A")
        wind      = cur.get("wind_speed_10m", "N/A")
        max_t     = (day.get("temperature_2m_max") or ["N/A"])[0]
        min_t     = (day.get("temperature_2m_min") or ["N/A"])[0]
        precip    = (day.get("precipitation_sum") or ["N/A"])[0]

        return (
            f"**Weather in {city_name}, {country}:**\n"
            f"- Current temperature: {temp}°C\n"
            f"- Wind speed: {wind} km/h\n"
            f"- Today's high: {max_t}°C\n"
            f"- Today's low: {min_t}°C\n"
            f"- Precipitation: {precip} mm"
        )

    except requests.exceptions.RequestException as e:
        return f"Weather API error: {str(e)}"
    except Exception as e:
        return f"Weather processing error: {str(e)}"


# ---------------------------------------------------------------------------
# Web search — Tavily (primary) with LLM fallback
# ---------------------------------------------------------------------------
def web_search_tool(query: str) -> str:
    """
    Web search tool using Tavily API.
    - Tavily is an AI-optimized search engine designed for LLM applications.
    - Falls back to LLM knowledge if Tavily is unavailable.
    """
    # Step 1: Try Tavily search
    tavily_result = _try_tavily(query)
    if tavily_result:
        return tavily_result

    # Step 2: Fallback to LLM knowledge if Tavily fails
    return _llm_fallback(query)


def _clean_snippet(text: str) -> str:
    """
    Strip markdown/HTML that would render as large headings or bold blocks
    when displayed in Streamlit. Keeps the text readable but plain.
    """
    # Remove markdown headings (# ## ### etc.)
    text = re.sub(r"(?m)^#{1,6}\s+", "", text)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple newlines into a single space
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _try_tavily(query: str) -> str:
    """
    Query Tavily and return a formatted response with answer + sources.
    Returns empty string on failure.
    """
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return ""

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="basic",
            max_results=5,
            include_answer=True,
        )

        lines = ["**🌐 Web Search Tool (Tavily)**\n"]

        # Tavily's AI-generated summary answer (clean it too)
        answer = response.get("answer", "")
        if answer:
            lines.append(f"{_clean_snippet(answer)}\n")

        # Source links — snippet is cleaned to avoid large rendered headings
        results = response.get("results", [])
        if results:
            lines.append("---\n**🔗 Sources:**")
            for r in results:
                title   = r.get("title", "").strip()
                url     = r.get("url", "").strip()
                content = r.get("content", "").strip()
                if title and url:
                    entry = f"- **{title}**  \n  {url}"
                    if content:
                        # Clean markdown/HTML, then truncate to 180 chars
                        clean = _clean_snippet(content)
                        snippet = clean[:180].rstrip() + ("…" if len(clean) > 180 else "")
                        entry += f"  \n  *{snippet}*"
                    lines.append(entry)

        return "\n".join(lines) if len(lines) > 1 else ""

    except Exception:
        return ""


def _llm_fallback(query: str) -> str:
    """Answer directly from LLM knowledge when Tavily is unavailable."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""You are a knowledgeable assistant. Answer the following question clearly,
accurately, and in a well-structured way. Use bullet points or numbered lists where appropriate.

Question: {query}"""
        answer = llm.invoke(prompt).content
        return f"**🌐 Web Search Tool (AI knowledge)**\n\n{answer}"
    except Exception as e:
        return f"Web search unavailable: {str(e)}"