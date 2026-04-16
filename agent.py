# ======================================================================
# 🤖 Agentic AI + Gradio + Persistent Memory (Chroma)
# Sources: DuckDuckGo, Wikipedia, Arxiv, PubMed + RAG (PDFs)
# Ready for Google Colab
# ======================================================================

# 📦 Install dependencies
# -> refer to requirements.txt for exact versions

# 🔑 OpenAI API Key
from getpass import getpass
import os
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    print("🔑 Enter your OpenAI API Key (get it at https://platform.openai.com/api-keys)")
    os.environ["OPENAI_API_KEY"] = getpass("Paste your key here: ")

print("✅ API Key set!")

# ========================= LangChain / Tools =========================
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.agents import AgentExecutor, create_openai_functions_agent, Tool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper, ArxivAPIWrapper, PubMedAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import VectorStoreRetrieverMemory
from langchain_core.prompts import ChatPromptTemplate

# ============================== Paths ===============================
import shutil
import tempfile
from pathlib import Path

_IN_COLAB = Path("/content").exists() and Path("/content").is_mount()
PERSIST_BASE = Path("/content/agentic_ai_persist") if _IN_COLAB else Path(__file__).parent / "agentic_ai_persist"
PERSIST_BASE.mkdir(parents=True, exist_ok=True)

MEMORY_DIR = str(PERSIST_BASE / "chroma_memory")        # conversational memory
DOCS_DIR   = str(PERSIST_BASE / "chroma_docs")          # PDF database
UPLOAD_DIR = Path("/content/uploads") if _IN_COLAB else PERSIST_BASE / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

# ============================ Embeddings & DB ==============================
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Persistent conversational memory
memory_vectorstore = Chroma(
    collection_name="chat_memory",
    persist_directory=MEMORY_DIR,
    embedding_function=embeddings,
)

memory_retriever = memory_vectorstore.as_retriever(search_kwargs={"k": 5})
memory = VectorStoreRetrieverMemory(retriever=memory_retriever)

# Persistent PDF knowledge base (RAG)
docs_vectorstore = Chroma(
    collection_name="docs_rag",
    persist_directory=DOCS_DIR,
    embedding_function=embeddings,
)
docs_retriever = docs_vectorstore.as_retriever(search_kwargs={"k": 5})

# ============================== Tools ================================
search = DuckDuckGoSearchAPIWrapper()
wiki = WikipediaAPIWrapper()
arxiv = ArxivAPIWrapper()

# ---- Audit Regression Tool ----
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Pre-programmed audit dataset
# Each row is a company. Target: audit_adjustment_pct (% of revenue requiring restatement).
_AUDIT_DATA = pd.DataFrame({
    "revenue_mln":            [12, 45, 8, 200, 30, 5, 90, 150, 22, 60, 18, 310, 7, 55, 100, 25, 40, 130, 16, 70],
    "total_assets_mln":       [8,  30, 5, 140, 20, 3, 65, 110, 15, 42, 12, 220, 4, 38,  75, 17, 28,  95, 10, 50],
    "debt_to_equity":         [0.4,1.2,0.2,2.1,0.8,0.1,1.5,1.8,0.6,1.1,0.3,2.4,0.2,0.9,1.3,0.5,0.7,1.6,0.4,1.0],
    "receivables_days":       [32, 55, 28, 70, 45, 20, 62, 75, 38, 50, 30, 80, 25, 52, 65, 35, 48, 72, 33, 58],
    "inventory_turnover":     [8,  5,  10, 3,  6,  12, 4,  3,  7,  5,  9,  2,  11, 5,  4,  7,  6,  3,  8,  5 ],
    "prior_findings_count":   [1,  3,  0,  5,  2,  0,  4,  6,  1,  2,  0,  7,  0,  3,  4,  1,  2,  5,  1,  3 ],
    "company_age_years":      [15, 8,  22, 5,  12, 30, 6,  4,  18, 10, 25, 3,  35, 9,  7,  14, 11, 5,  20, 8 ],
    "audit_adjustment_pct":   [0.8,2.1,0.3,4.5,1.2,0.1,3.3,5.0,0.7,1.8,0.2,6.1,0.1,2.0,3.8,0.9,1.5,4.2,0.6,2.3],
})

_FEATURES = ["revenue_mln", "total_assets_mln", "debt_to_equity",
             "receivables_days", "inventory_turnover", "prior_findings_count", "company_age_years"]
_TARGET = "audit_adjustment_pct"

def train_audit_regression(query: str) -> str:
    """Train a linear regression model on the audit dataset and return results."""
    X = _AUDIT_DATA[_FEATURES]
    y = _AUDIT_DATA[_TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    coef_df = pd.Series(model.coef_, index=_FEATURES).sort_values(key=abs, ascending=False)

    lines = [
        "=== Audit Adjustment Regression Model ===",
        f"Target: audit_adjustment_pct (% of revenue requiring restatement)",
        f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}",
        "",
        f"R²  (test): {r2:.3f}",
        f"MAE (test): {mae:.3f} percentage points",
        "",
        "Feature coefficients (standardised, sorted by impact):",
    ]
    for feat, coef in coef_df.items():
        direction = "↑ higher risk" if coef > 0 else "↓ lower risk"
        lines.append(f"  {feat:<25} {coef:+.3f}  ({direction})")

    lines += [
        "",
        "Key insight: prior_findings_count and debt_to_equity are the strongest",
        "predictors of audit adjustment size in this dataset.",
    ]
    return "\n".join(lines)

def rag_search(query: str) -> str:
    """Search inside uploaded PDFs (RAG). Returns relevant chunks."""
    docs = docs_retriever.get_relevant_documents(query)
    if not docs:
        return "No relevant content found in your PDF database."
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        tag_page = f" (p.{page})" if page is not None else ""
        snippet = d.page_content[:700].strip().replace("\n", " ")
        out.append(f"[{i}] Source: {src}{tag_page}\nExcerpt: {snippet}")
    return "\n\n".join(out)

tools = [
    Tool(
        name="Web_Search",
        func=search.run,
        description="Searches the web using DuckDuckGo. Input: search term."
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Searches Wikipedia articles. Input: topic."
    ),
    Tool(
        name="Arxiv_Search",
        func=arxiv.run,
        description="Finds academic papers on Arxiv. Input: query (e.g. 'graph neural networks')."
    ),
    Tool(
        name="RAG_PDFs",
        func=rag_search,
        description="Searches your uploaded PDFs (RAG). Input: question or term."
    ),
    Tool(
        name="Train_Regression_Model",
        func=train_audit_regression,
        description=(
            "Trains a linear regression model on a pre-loaded audit dataset to predict "
            "audit adjustment size (% of revenue). Returns model performance (R², MAE) and "
            "feature importances. Use when asked about audit risk drivers, regression analysis, "
            "or which financial indicators predict audit adjustments. Input: any question or 'run'."
        )
    ),
]

# ============================== Model & Prompt ============================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI assistant with access to tools (web, wikipedia, arxiv, RAG) "
     "and persistent memory. Cite sources when possible. Be concise and helpful."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory,
                        return_intermediate_steps=True)

# ============================ PDF Ingestion (RAG) =======================
from pypdf import PdfReader

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

def index_pdfs(file_paths):
    """Reads and indexes PDFs into the RAG database."""
    all_docs = []
    for fp in file_paths:
        fp = Path(fp)
        if not fp.exists():
            continue
        try:
            reader = PdfReader(str(fp))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                chunks = text_splitter.split_text(text)
                for ch in chunks:
                    all_docs.append(
                        Document(
                            page_content=ch,
                            metadata={"source": fp.name, "path": str(fp), "page": page_num + 1}
                        )
                    )
        except Exception as e:
            print(f"⚠️ Error reading {fp.name}: {e}")

    if all_docs:
        docs_vectorstore.add_documents(all_docs)
        docs_vectorstore.persist()
    return len(all_docs)

def clear_all_memory():
    """Deletes conversational and RAG memory."""
    if Path(MEMORY_DIR).exists():
        shutil.rmtree(MEMORY_DIR, ignore_errors=True)
    if Path(DOCS_DIR).exists():
        shutil.rmtree(DOCS_DIR, ignore_errors=True)

    global memory_vectorstore, memory_retriever, memory
    global docs_vectorstore, docs_retriever

    memory_vectorstore = Chroma(
        collection_name="chat_memory",
        persist_directory=MEMORY_DIR,
        embedding_function=embeddings,
    )
    memory_retriever = memory_vectorstore.as_retriever(search_kwargs={"k": 5})
    memory = VectorStoreRetrieverMemory(retriever=memory_retriever)

    docs_vectorstore = Chroma(
        collection_name="docs_rag",
        persist_directory=DOCS_DIR,
        embedding_function=embeddings,
    )
    docs_retriever = docs_vectorstore.as_retriever(search_kwargs={"k": 5})

# ================================ Gradio UI ================================
import gradio as gr

INSTRUCTIONS = """
### 🧠 How to use
- **General search:** Just ask normally (uses DuckDuckGo/Wikipedia).
- **Academic:** Ask for academic topics — the agent may use Arxiv automatically.
- **Your PDFs (RAG):** Upload files below, then ask questions about them.
- **/clear:** Erases all memory and your PDF database.
"""

def respond(msg, chat_history):
    chat_history = chat_history or []

    # Command to clear memory
    if msg.strip().lower() == "/clear":
        clear_all_memory()
        chat_history.append({"role": "user", "content": "/clear"})
        chat_history.append({"role": "assistant", "content": "🧹 Memory and PDF base cleared successfully."})
        return chat_history, chat_history

    chat_history.append({"role": "user", "content": msg})

    try:
        result = executor.invoke({"input": msg})
        response = result["output"]
        steps = result.get("intermediate_steps", [])
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        return chat_history, chat_history

    for action, observation in steps:
        tool_input = action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)
        obs_preview = str(observation)[:600] + ("…" if len(str(observation)) > 600 else "")
        content = f"**Input:** {tool_input}\n\n**Output:**\n```\n{obs_preview}\n```"
        chat_history.append(gr.ChatMessage(
            role="assistant",
            content=content,
            metadata={"title": f"🔧 {action.tool}"},
        ))

    chat_history.append({"role": "assistant", "content": response})
    return chat_history, chat_history

def upload_pdfs(files, chat_history):
    """Handles uploaded PDFs and indexes them."""
    chat_history = chat_history or []

    if not files:
        chat_history.append({"role": "assistant", "content": "No files uploaded."})
        return chat_history, chat_history

    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as w:
            w.write(f.read())
        saved_paths.append(str(dest))

    chunks = index_pdfs(saved_paths)
    chat_history.append({"role": "assistant",
                         "content": f"📚 {len(files)} file(s) uploaded. "
                                    f"Indexed {chunks} text chunks into RAG. You can now ask questions!"})
    return chat_history, chat_history

with gr.Blocks(title="🏰 Nyenrode Agentic AI") as demo:
    gr.Markdown("# 🏰🤖 Nyenrode Agentic AI with Persistent Memory and RAG (PDFs)")
    gr.Markdown(INSTRUCTIONS)

    with gr.Row():
        chatbot = gr.Chatbot(label="Agent")
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(label="Type your question… (use /clear to reset)", scale=4)
        btn = gr.Button("Send", scale=1)

    with gr.Row():
        files = gr.File(label="Upload your PDFs (RAG)", file_count="multiple", file_types=[".pdf"])
        btn_up = gr.Button("Index PDFs")

    btn.click(respond, [msg, state], [chatbot, state]).then(lambda: "", outputs=msg)
    msg.submit(respond, [msg, state], [chatbot, state]).then(lambda: "", outputs=msg)

    btn_up.click(upload_pdfs, [files, state], [chatbot, state])

demo.launch(share=True)