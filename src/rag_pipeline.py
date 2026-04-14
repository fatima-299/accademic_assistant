import os
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from src.vector_store import load_vector_store

# Module-level LangChain memory — persists for the lifetime of the Python process
# (i.e. the entire Streamlit session). Keeps the last 6 turns.
_rag_memory = ConversationBufferWindowMemory(
    k=6,
    memory_key="chat_history",
    return_messages=False,
    human_prefix="User",
    ai_prefix="Assistant",
)


def reset_rag_memory():
    """Wipe RAG conversational memory (call from a clear-chat button if needed)."""
    _rag_memory.clear()


# Sentinel phrase the LLM writes when documents don't cover the query
_RAG_NOT_FOUND = "the documents do not contain enough information"


def rag_has_answer(text: str) -> bool:
    """Return False if the RAG response is the 'not found' sentinel."""
    return _RAG_NOT_FOUND not in text.lower()


def answer_with_rag(query: str, history: str = "") -> str:
    """
    Retrieve relevant chunks from FAISS, then generate a grounded answer.

    Citations are appended at the end of every response in a user-visible block:
        Sources used:
          [1] machineLearning.pdf — page 12
          [2] ANNs.pdf — page 5
    """
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)

    # Build LLM context and a pre-formatted citation block
    context_parts = []
    source_lines = []
    for i, doc in enumerate(docs, start=1):
        raw_source = doc.metadata.get("source", "unknown")
        filename = os.path.basename(raw_source)
        page = doc.metadata.get("page", "N/A")
        # PyPDFLoader returns 0-indexed page numbers; display as 1-indexed
        try:
            page_display = int(page) + 1
        except (TypeError, ValueError):
            page_display = page

        context_parts.append(
            f"[Source {i} | {filename} | page {page_display}]\n{doc.page_content}"
        )
        source_lines.append(f"  [{i}] {filename} — page {page_display}")

    context = "\n\n".join(context_parts)
    # This block will be injected verbatim at the end of the answer
    sources_block = "---\n**Sources used:**\n" + "\n".join(source_lines)

    # Prefer the LangChain memory buffer over the raw history string passed from app.py
    lc_history = _rag_memory.load_memory_variables({}).get("chat_history", "")
    effective_history = lc_history if lc_history.strip() else history

    prompt = f"""You are an academic assistant specialized in machine learning and neural networks.

Use the conversation history to understand follow-up questions such as:
"explain it more simply", "give me 3 key points", "what is the difference between both".

Answer ONLY from the document context provided below.
If the answer is not in the context, say clearly:
"The documents do not contain enough information to answer this question."

Conversation history:
{effective_history}

Document context:
{context}

User question:
{query}

Instructions:
- Give a clear, well-structured answer.
- Use the conversation history for follow-up questions.
- Do NOT invent information not present in the context.
- At the very end of your answer, output EXACTLY the following block (copy it verbatim, do not change it):
{sources_block}
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)
    answer = response.content

    # Only save to memory if RAG actually answered — don't pollute memory
    # with "not found" responses so follow-up questions still work correctly.
    if rag_has_answer(answer):
        _rag_memory.save_context({"input": query}, {"output": answer})

    return answer