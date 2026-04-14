import streamlit as st
from dotenv import load_dotenv
from src.agent import route_query
from src.rag_pipeline import reset_rag_memory

load_dotenv()

st.set_page_config(
    page_title="Intelligent Academic Assistant",
    page_icon="🎓",
    layout="wide",
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main { padding-top: 1rem; }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1d4ed8);
        padding: 2rem 2rem 1.5rem 2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    .hero h1  { margin: 0; font-size: 2.3rem; font-weight: 800; }
    .hero p   { margin-top: 0.6rem; font-size: 1.05rem; opacity: 0.95; }

    .feature-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }
    .feature-card h4 { margin: 0 0 0.4rem 0; font-size: 1rem; }
    .feature-card p  { margin: 0; color: #334155; font-size: 0.95rem; }

    .small-note { color: #475569; font-size: 0.9rem; }

    .example-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
    }

    .footer-note {
        margin-top: 1.5rem;
        color: #64748b;
        font-size: 0.9rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### 🔑 OpenAI API")
    st.markdown(
        "<div class='small-note'>API key configured in the <code>.env</code> file.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("## 📘 User Guide")
    st.markdown("### Query types")
    st.markdown("""
- **Course questions (RAG)**
  Example: *What is supervised learning?*

- **Calculations**
  Example: *calculate sqrt(36) + sin(pi/2)*

- **Weather**
  Example: *What is the weather in Paris?*

- **Web search**
  Example: *What is the Mamba architecture?*

- **Task management**
  Example: *add task: revise neural networks*
  Example: *show tasks*
  Example: *delete task: 2*
  Example: *clear tasks*
""")

    st.markdown("---")
    st.markdown("## ✅ Test examples")
    st.markdown("""
- `hello`
- `what is supervised learning?`
- `explain it more simply`
- `give me 3 key points`
- `what is gradient descent?`
- `calculate 12*5+6`
- `calculate sqrt(144) + log(100,10)`
- `add task: revise neural networks`
- `show tasks`
- `delete task: 1`
- `what is the weather in London?`
- `What is the Mamba architecture in deep learning?`
- `What is LangChain used for?`
""")

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        reset_rag_memory()
        st.rerun()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🎓 Intelligent Academic Assistant</h1>
    <p>RAG · AI Agents · Conversational Memory · Calculations · Weather · Web Search</p>
</div>
""", unsafe_allow_html=True)

# ---------- Top Info Section ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Welcome")
    st.write(
        "Ask questions about your academic documents (ML, deep learning, neural networks), "
        "perform scientific calculations, manage your task list, get real-time weather, "
        "or search the web — all in one interface."
    )

with col2:
    st.markdown("### Features")
    st.markdown("""
<div class="feature-card">
    <h4>📚 RAG on your documents</h4>
    <p>Answers with citations (filename + page number).</p>
</div>
<div class="feature-card">
    <h4>🧮 Scientific calculator</h4>
    <p>sqrt, sin, cos, log, factorial and more.</p>
</div>
<div class="feature-card">
    <h4>📝 Persistent to-do list</h4>
    <p>Tasks saved between sessions.</p>
</div>
<div class="feature-card">
    <h4>🌤️ Real-time weather</h4>
    <p>Via the Open-Meteo API (free, no key needed).</p>
</div>
<div class="feature-card">
    <h4>🌐 Web search</h4>
    <p>Tavily AI search — accurate and up-to-date results.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Example Prompts ----------
with st.expander("💡 Query suggestions", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='example-box'>what is supervised learning?</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>what is a perceptron?</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>explain it more simply</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>give me 3 key points</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='example-box'>calculate sqrt(36) + sin(pi/2)</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>add task: revise chapter 2</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>what is the weather in London?</div>", unsafe_allow_html=True)
        st.markdown("<div class='example-box'>What is the Mamba architecture?</div>", unsafe_allow_html=True)

# ---------- Chat ----------
st.markdown("### 💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    recent = st.session_state.messages[-40:]  # full session for recap
    history_parts = []
    for msg in recent[:-1]:
        role = msg["role"].capitalize()
        content = msg["content"]
        # For assistant messages, keep first 200 chars — enough to identify the topic
        # but short enough to stay within LLM context window across 40 messages
        if msg["role"] == "assistant" and len(content) > 200:
            content = content[:200].rstrip() + "..."
        history_parts.append(f"{role}: {content}")
    history = "\n".join(history_parts)

    with st.spinner("Thinking…"):
        response = route_query(user_input, history)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# ---------- Footer ----------
st.markdown(
    "<div class='footer-note'>"
    "Academic Assistant — RAG + AI Agents + Conversational Memory (LangChain) + Streamlit"
    "</div>",
    unsafe_allow_html=True,
)