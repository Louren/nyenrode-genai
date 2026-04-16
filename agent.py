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
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, Tool
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

# ---- DigiJazz Revenue Regression Tool ----
import pandas as pd
from pathlib import Path as _Path
import statsmodels.api as _sm

# Load pre-generated DigiJazz weekly data (or generate on the fly if CSV missing)
_DATA_CSV = _Path(__file__).parent / "digijazz_data.csv"

def _load_digijazz_data() -> pd.DataFrame:
    if _DATA_CSV.exists():
        return pd.read_csv(_DATA_CSV, parse_dates=["week"])
    # Fallback: generate in memory
    from generate_demo_data import generate
    return generate()

_ALL_COST_FEATURES = [
    "marketing_expenses",
    "it_costs",
    "shipping_costs",
    "employee_expenses",
    "rental_costs",
    "legal_costs",
    "lease_car_costs",
    "grocery_costs",
]
_TARGET = "revenue"

def digijazz_dataset_info(_: str) -> str:
    """Return a summary of the DigiJazz dataset."""
    df = _load_digijazz_data()
    return (
        f"DigiJazz webshop dataset — {len(df)} weekly observations "
        f"({df['week'].min().date()} to {df['week'].max().date()}).\n"
        f"Target variable: revenue (weekly €).\n"
        f"Available cost predictors: {', '.join(_ALL_COST_FEATURES)}.\n\n"
        + df[[_TARGET] + _ALL_COST_FEATURES].describe().to_string()
    )


def digijazz_list_features(_: str) -> str:
    """List available cost features with basic statistics."""
    df = _load_digijazz_data()
    lines = [f"{'Feature':<22}  {'Mean (€)':>12}  {'Std (€)':>11}  {'Min (€)':>11}  {'Max (€)':>11}"]
    lines.append("-" * 72)
    for feat in _ALL_COST_FEATURES:
        col = df[feat]
        lines.append(f"{feat:<22}  {col.mean():>12,.0f}  {col.std():>11,.0f}  {col.min():>11,.0f}  {col.max():>11,.0f}")
    return "\n".join(lines)


def digijazz_train_model(features: str) -> str:
    """Train OLS on the given comma-separated feature names and return the full results."""
    df = _load_digijazz_data()

    selected = [f.strip() for f in features.split(",") if f.strip() in _ALL_COST_FEATURES]
    if not selected:
        return f"No valid features found. Available: {', '.join(_ALL_COST_FEATURES)}"

    n_test = 26
    train_df, test_df = df.iloc[:-n_test], df.iloc[-n_test:]

    result = _sm.OLS(train_df[_TARGET], _sm.add_constant(train_df[selected])).fit()

    y_pred = result.predict(_sm.add_constant(test_df[selected]))
    y_test = test_df[_TARGET]
    oos_r2   = 1 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()
    oos_mae  = (y_test - y_pred).abs().mean()
    oos_rmse = ((y_test - y_pred) ** 2).mean() ** 0.5

    return (
        result.summary().as_text()
        + f"\n\nOut-of-sample (last {n_test} weeks):\n"
        + f"  R²   {oos_r2:.4f}  {'(good fit)' if oos_r2 > 0.7 else '(weak)'}\n"
        + f"  MAE  €{oos_mae:,.0f}\n"
        + f"  RMSE €{oos_rmse:,.0f}"
    )

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
        name="DigiJazz_Dataset_Info",
        func=digijazz_dataset_info,
        description="Returns an overview and descriptive statistics of the DigiJazz weekly dataset. Use this when the user asks about the data, its time range, or wants a general summary. Input: anything (ignored)."
    ),
    Tool(
        name="DigiJazz_List_Features",
        func=digijazz_list_features,
        description="Lists all available cost features for the DigiJazz revenue regression model with their mean, std, min, and max. Use this to help the user decide which features to include before training. Input: anything (ignored)."
    ),
    Tool(
        name="DigiJazz_Train_Model",
        func=digijazz_train_model,
        description="Trains an OLS regression to predict DigiJazz weekly revenue. Input: comma-separated feature names to include, e.g. 'marketing_expenses, it_costs, shipping_costs'. Available features: marketing_expenses, it_costs, shipping_costs, employee_expenses, rental_costs, legal_costs, lease_car_costs, grocery_costs."
    ),
]

# ============================== Model & Prompt ============================
AVAILABLE_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-4.1-mini"]
DEFAULT_MODEL = "gpt-5.4-mini"

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI assistant with access to tools (web, wikipedia, arxiv, RAG) "
     "and persistent memory. Cite sources when possible. Be concise and helpful."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

_executor_cache: dict = {}

def _get_executor(model_name: str) -> AgentExecutor:
    if model_name not in _executor_cache:
        llm = ChatOpenAI(model=model_name, temperature=0.3)
        agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
        _executor_cache[model_name] = AgentExecutor(
            agent=agent, tools=tools, verbose=True, memory=memory,
            return_intermediate_steps=True,
        )
    return _executor_cache[model_name]

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
import json

CHAT_LOG = str(PERSIST_BASE / "chat_history.json")

def _load_chat_history():
    try:
        with open(CHAT_LOG, "r") as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    # Restore tool-call messages as ChatMessage objects so Gradio accepts them
    restored = []
    for m in raw:
        if isinstance(m, dict) and m.get("metadata"):
            restored.append(gr.ChatMessage(role=m["role"], content=m["content"],
                                           metadata=m["metadata"]))
        else:
            restored.append(m)
    return restored

def _save_chat_history(chat_history):
    # gr.ChatMessage objects are not JSON-serialisable; convert to plain dicts first
    serialisable = []
    for msg in chat_history:
        if isinstance(msg, gr.ChatMessage):
            serialisable.append({"role": msg.role, "content": msg.content,
                                  "metadata": msg.metadata})
        else:
            serialisable.append(msg)
    with open(CHAT_LOG, "w") as f:
        json.dump(serialisable, f, indent=2)

INSTRUCTIONS = """
### 🧠 How to use
- **General search:** Just ask normally (uses DuckDuckGo/Wikipedia).
- **Academic:** Ask for academic topics — the agent may use Arxiv automatically.
- **Your PDFs (RAG):** Upload files below, then ask questions about them.
- **/clear:** Erases all memory and your PDF database.
"""

def respond(msg, chat_history, model_name):
    chat_history = chat_history or []

    # Command to clear memory
    if msg.strip().lower() == "/clear":
        clear_all_memory()
        chat_history.append({"role": "user", "content": "/clear"})
        chat_history.append({"role": "assistant", "content": "🧹 Memory and PDF base cleared successfully."})
        _save_chat_history(chat_history)
        return chat_history, chat_history

    chat_history.append({"role": "user", "content": msg})

    try:
        result = _get_executor(model_name).invoke({"input": msg})
        response = result["output"]
        steps = result.get("intermediate_steps", [])
    except Exception as e:
        chat_history.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        _save_chat_history(chat_history)
        return chat_history, chat_history

    for action, observation in steps:
        tool_input = action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)
        content = f"**Input:** {tool_input}\n\n**Output:**\n```\n{observation}\n```"
        chat_history.append(gr.ChatMessage(
            role="assistant",
            content=content,
            metadata={"title": f'The LLM used the tool "{action.tool}"', "status": "done"},
        ))

    chat_history.append({"role": "assistant", "content": response})
    _save_chat_history(chat_history)
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
    _save_chat_history(chat_history)
    return chat_history, chat_history

with gr.Blocks(title="🏰 Nyenrode Agentic AI") as demo:
    gr.Markdown("# 🏰🤖 Nyenrode Agentic AI with Persistent Memory and RAG (PDFs)")
    gr.Markdown(INSTRUCTIONS)

    with gr.Row():
        chatbot = gr.Chatbot(label="Agent")
    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(label="Type your question… (use /clear to reset)", scale=4)
        model_selector = gr.Dropdown(
            choices=AVAILABLE_MODELS, value=DEFAULT_MODEL, label="Model", scale=1
        )
        btn = gr.Button("Send", scale=1)

    with gr.Row():
        files = gr.File(label="Upload your PDFs (RAG)", file_count="multiple", file_types=[".pdf"])
        btn_up = gr.Button("Index PDFs")

    def _on_load():
        h = _load_chat_history()
        return h, h

    demo.load(_on_load, outputs=[chatbot, state])

    btn.click(respond, [msg, state, model_selector], [chatbot, state]).then(lambda: "", outputs=msg)
    msg.submit(respond, [msg, state, model_selector], [chatbot, state]).then(lambda: "", outputs=msg)

    btn_up.click(upload_pdfs, [files, state], [chatbot, state])

demo.launch(share=True)