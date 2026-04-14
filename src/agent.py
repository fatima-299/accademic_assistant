"""
Agent — decides which tool / pipeline handles a user query.

This is the core agent of the system (Partie 3 of the project).
It receives every user message and autonomously selects the right action:
  - RAG pipeline   → academic questions from local documents
  - Calculator     → mathematical expressions
  - Weather tool   → weather queries
  - Todo tool      → task list management
  - Web search     → current/online information (Tavily)
  - Direct chat    → casual conversation

Strategy (ordered, first match wins):
  1. Keyword pre-filter  — O(1), zero LLM cost for obvious queries
  2. LLM classifier      — gpt-4o-mini, only reached for ambiguous inputs
"""

import json
import re
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from src.rag_pipeline import answer_with_rag, rag_has_answer
from src.tools import calculator_tool, todo_tool, weather_tool, web_search_tool

# ---------------------------------------------------------------------------
# Conversational memory for non-RAG chat turns
# ---------------------------------------------------------------------------
_chat_memory = ConversationBufferWindowMemory(
    k=6,
    memory_key="chat_history",
    return_messages=False,
    human_prefix="User",
    ai_prefix="Assistant",
)

# ---------------------------------------------------------------------------
# Keyword pre-filter patterns
# ---------------------------------------------------------------------------

# Follow-up phrases that always belong to RAG — checked FIRST before anything else
# so they are never accidentally routed to the calculator or LLM router.
_RAG_FOLLOWUP_RE = re.compile(
    r"^\s*(explain\s+(it|this|that|more|again|further)|"
    r"(explain|describe|elaborate)\s+(it|this|that)?\s*(more|further|simply|clearly|briefly|in\s+detail|again)?|"
    r"give\s+me\s+\d+\s+key\s+points?|"
    r"give\s+me\s+(a\s+)?(summary|overview|example|examples)|"
    r"what\s+(is\s+)?(the\s+)?(difference|relation|link|connection)\s+(between|with)?|"
    r"compare\s+(the\s+)?(two|both|them)|"
    r"(can\s+you\s+)?(simplif(y|ied)|restate|rephrase|clarif(y|ied))|"
    r"(tell|show)\s+me\s+more|"
    r"what\s+do\s+you\s+mean|"
    r"how\s+(does|do|did)\s+(it|this|that)\s+work|"
    r"why\s+(is|does|do)\s+(it|this|that)|"
    r"give\s+me\s+(more\s+)?(details?|info|information))\b",
    re.IGNORECASE,
)

# Calculator — only matches when the expression starts with a math keyword
# or begins directly with digits/parentheses followed by an operator.
# NOTE: does NOT match plain English sentences like "explain it more simply".
_CALC_RE = re.compile(
    r"^\s*(calculate|compute|eval(uate)?\s|"
    r"sqrt\s*\(|sin\s*\(|cos\s*\(|tan\s*\(|log\s*\(|exp\s*\(|factorial\s*\(|"
    r"[\d(][\d\s\(\)\.]*[\+\-\*\/\^])",
    re.IGNORECASE,
)

_TODO_RE = re.compile(
    r"(add\s+task|show\s+tasks?|list\s+tasks?|display\s+tasks?|"
    r"delete\s+task|remove\s+task|clear\s+tasks?)",
    re.IGNORECASE,
)

_WEATHER_RE = re.compile(
    r"(weather|temperature|forecast|how\s+(hot|cold|warm)|what.s\s+the\s+(temp|climate))",
    re.IGNORECASE,
)

_CHAT_RE = re.compile(
    r"^\s*(hi|hello|hey|bonjour|salut|thanks?|thank\s+you|merci|"
    r"good\s+(morning|afternoon|evening|night)|how\s+are\s+you|"
    r"what\s+(can|do)\s+you\s+do|help\s+me|who\s+are\s+you|"
    r"what\s+have\s+we\s+(talked|discussed|covered|spoken)\s+(about)?|"
    r"(summarize|summary\s+of)\s+(our|this|the)\s+(conversation|chat|session)|"
    r"what\s+did\s+we\s+talk\s+about|"
    r"go\s+back\s+to\s+what\s+you\s+said|"
    r"remind\s+me\s+(what|about)|"
    r"what\s+was\s+(the|your)\s+(last|previous|earlier)\s+(answer|response|point)|"
    r"can\s+you\s+recap|"
    r"earlier\s+you\s+(said|mentioned|told))\b",
    re.IGNORECASE,
)

# Academic / RAG topic keywords
_RAG_KEYWORDS = re.compile(
    r"\b(supervised|unsupervised|machine\s+learning|deep\s+learning|neural\s+network|"
    r"perceptron|gradient\s+descent|backprop|regression|classification|clustering|"
    r"overfitting|underfitting|activation\s+function|loss\s+function|"
    r"transformer|attention\s+mechanism|cnn|rnn|lstm|reinforcement|"
    r"training|dataset|embedding|vector|epoch|batch|dropout)\b",
    re.IGNORECASE,
)


def _keyword_route(query: str):
    """Return (action, clean_input) or None if no pattern matches."""
    q = query.strip()

    # RAG follow-ups MUST be checked before the calculator to avoid misrouting
    # phrases like "explain it more simply" or "give me 3 key points".
    if _RAG_FOLLOWUP_RE.match(q):
        return ("rag", q)

    if _TODO_RE.search(q):
        return ("todo", q)

    if _CALC_RE.match(q):
        return ("calculator", q)

    if _WEATHER_RE.search(q):
        # Pass the full query to weather_tool — it handles all cleaning internally
        return ("weather", q)

    if _CHAT_RE.match(q):
        return ("chat", q)

    if _RAG_KEYWORDS.search(q):
        return ("rag", q)

    return None  # fall through to LLM router


# ---------------------------------------------------------------------------
# LLM router — called only when keyword pre-filter has no match
# ---------------------------------------------------------------------------
def _llm_route(query: str, history: str):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    router_prompt = f"""You are a routing agent for an academic assistant.

Choose exactly ONE action for the user's question.

Available actions:
- "rag"        -> questions about academic topics (machine learning, neural networks, etc.)
- "calculator" -> mathematical expressions and calculations
- "todo"       -> task list management (add, show, delete tasks)
- "weather"    -> weather queries for any city
- "web"        -> recent/current/online information not in academic documents
- "chat"       -> casual conversation, greetings, off-topic questions

Conversation history:
{history}

User question: {query}

Return ONLY valid JSON (no markdown, no explanation):
{{"action": "rag|calculator|todo|weather|web|chat", "input": "clean input for the tool"}}
"""

    try:
        raw = llm.invoke(router_prompt).content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        route = json.loads(raw)
        action = route.get("action", "chat").lower()
        tool_input = route.get("input", query).strip()
        return action, tool_input
    except Exception:
        return "chat", query.strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def route_query(query: str, history: str = "") -> str:
    """
    Route a user query to the right handler and return the response string.
    Uses the LangChain ConversationBufferWindowMemory for chat turns.
    """
    # 1. Try keyword pre-filter first (free)
    result = _keyword_route(query)
    if result:
        action, tool_input = result
    else:
        # 2. Fall back to LLM router
        action, tool_input = _llm_route(query, history)

    # Dispatch
    if action == "rag":
        rag_answer = answer_with_rag(tool_input, history)
        # If RAG could not find the answer in the documents, fall back to web search
        # so the user gets a real answer instead of a dead end.
        if not rag_has_answer(rag_answer):
            # Documents don't cover this — get answer from web/AI instead
            return web_search_tool(tool_input)
        return rag_answer

    if action == "calculator":
        return calculator_tool(tool_input)

    if action == "todo":
        return todo_tool(tool_input)

    if action == "weather":
        return weather_tool(tool_input)

    if action == "web":
        return web_search_tool(tool_input)

    # "chat" — use the full session history from app.py (contains ALL turns:
    # RAG answers, calculations, weather, tasks, web searches, etc.)
    # lc_history only stores chat turns so we prefer the full history string.
    lc_history = _chat_memory.load_memory_variables({}).get("chat_history", "")

    # For recap queries always use the full app history; for normal chat use
    # whichever source is richer.
    # Detect recap intent using PRECISE phrases only — avoids false positives
    # like "how can you help me" or "go back to deep learning"
    recap_phrases = [
        "what have we talked", "what have we discussed", "what have we covered",
        "what did we talk", "what did we discuss",
        "what have we taled",   # common typo
        "what we have talked", "what we have discussed",
        "what we have taled",   # common typo
        "summarize our conversation", "summarize the conversation",
        "summary of our conversation", "recap our conversation",
        "what topics have we", "what subjects have we",
        "so far in our conversation",
    ]
    q_lower = query.lower()
    is_recap = any(phrase in q_lower for phrase in recap_phrases)

    if is_recap:
        # Use the full session history from app.py — contains ALL turns
        effective_history = history
    else:
        # Normal chat — prefer LangChain memory, fall back to app history
        effective_history = lc_history if lc_history.strip() else history

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)
    chat_prompt = f"""You are a helpful academic assistant.

Conversation history:
{effective_history}

User: {query}

Instructions:
- Reply naturally and helpfully to what the user is actually asking.
- ONLY provide a conversation summary if the user EXPLICITLY asks what topics
  have been discussed or talked about (e.g. "what have we talked about so far").
- For ALL other questions (greetings, capability questions, topic questions),
  answer directly and normally — do NOT give a summary.
- If the user asks to go back to a specific topic, find it in the history and
  continue from there.
- If you do not know something, say so honestly.
"""
    response = llm.invoke(chat_prompt).content
    _chat_memory.save_context({"input": query}, {"output": response})
    return response