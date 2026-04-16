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
from langchain_core.messages import HumanMessage
from langchain_classic.agents import AgentExecutor, create_openai_tools_agent, Tool
from langchain_core.tools import StructuredTool
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, WikipediaAPIWrapper, ArxivAPIWrapper, PubMedAPIWrapper
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ============================== Paths ===============================
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

# Conversational memory (buffer — maintains sequential message history)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

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

def digijazz_dataset_info() -> str:
    """Return a summary of the DigiJazz dataset."""
    df = _load_digijazz_data()
    return (
        f"DigiJazz webshop dataset — {len(df)} weekly observations "
        f"({df['week'].min().date()} to {df['week'].max().date()}).\n"
        f"Target variable: revenue (weekly €).\n"
        f"Available cost predictors: {', '.join(_ALL_COST_FEATURES)}.\n\n"
        + df[[_TARGET] + _ALL_COST_FEATURES].describe().to_string()
    )


def digijazz_list_features() -> str:
    """List available cost features with basic statistics."""
    df = _load_digijazz_data()
    lines = [f"{'Feature':<22}  {'Mean (€)':>12}  {'Std (€)':>11}  {'Min (€)':>11}  {'Max (€)':>11}"]
    lines.append("-" * 72)
    for feat in _ALL_COST_FEATURES:
        col = df[feat]
        lines.append(f"{feat:<22}  {col.mean():>12,.0f}  {col.std():>11,.0f}  {col.min():>11,.0f}  {col.max():>11,.0f}")
    return "\n".join(lines)


def digijazz_chart() -> str:
    """Generate a chart of all DigiJazz columns over time and return a base64-encoded PNG."""
    import base64, io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = _load_digijazz_data()

    fig, (ax_rev, ax_costs) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("DigiJazz — Weekly Financial Data", fontsize=14)

    ax_rev.plot(df["week"], df["revenue"], color="steelblue", linewidth=1.5)
    ax_rev.set_ylabel("Revenue (€)")
    ax_rev.set_title("Revenue")
    ax_rev.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    for col in _ALL_COST_FEATURES:
        ax_costs.plot(df["week"], df[col], label=col, linewidth=1)
    ax_costs.set_ylabel("Cost (€)")
    ax_costs.set_title("Cost Drivers")
    ax_costs.legend(fontsize=7, ncol=2)
    ax_costs.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    fig.autofmt_xdate()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"CHART_B64:{b64}"


def digijazz_train_model(features: str) -> str:
    """Train OLS on the given comma-separated feature names and return the full results."""
    df = _load_digijazz_data()

    selected = [f.strip() for f in features.split(",") if f.strip() in _ALL_COST_FEATURES]
    if not selected:
        return f"No valid features found. Available: {', '.join(_ALL_COST_FEATURES)}"

    n_test = 26
    df = df.copy()
    df["time"] = range(len(df))   # week index: captures the revenue trend

    train_df, test_df = df.iloc[:-n_test], df.iloc[-n_test:]

    predictors = ["time"] + selected
    result = _sm.OLS(train_df[_TARGET], _sm.add_constant(train_df[predictors])).fit()

    y_pred = result.predict(_sm.add_constant(test_df[predictors]))
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
    StructuredTool.from_function(
        func=digijazz_chart,
        name="DigiJazz_Chart",
        description="Generates a chart of all DigiJazz columns (revenue + all cost drivers) over time and displays it.",
    ),
    StructuredTool.from_function(
        func=digijazz_dataset_info,
        name="DigiJazz_Dataset_Info",
        description="Returns an overview and descriptive statistics of the DigiJazz weekly dataset. Use this when the user asks about the data, its time range, or wants a general summary.",
    ),
    StructuredTool.from_function(
        func=digijazz_list_features,
        name="DigiJazz_List_Features",
        description="Lists all available cost features for the DigiJazz revenue regression model with their mean, std, min, and max. Use this to help the user decide which features to include before training.",
    ),
    StructuredTool.from_function(
        func=digijazz_train_model,
        name="DigiJazz_Train_Model",
        description="Trains an OLS regression to predict DigiJazz weekly revenue. Input: comma-separated feature names to include, e.g. 'marketing_expenses, it_costs, shipping_costs'. Available features: marketing_expenses, it_costs, shipping_costs, employee_expenses, rental_costs, legal_costs, lease_car_costs, grocery_costs.",
    ),
]

# ============================== Model & Prompt ============================
AVAILABLE_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-4.1-mini"]
DEFAULT_MODEL = "gpt-5.4-mini"

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI audit assistant supporting an auditor working on the DigiJazz.nl engagement. "
     "DigiJazz was founded in 2015 by Dean Mus and Rob Tone and operates as a premier platform for jazz music lovers worldwide, "
     "offering streaming, downloads, and vinyl record sales. It expanded into vinyl in 2017. "
     "Your role is to help the auditor analyse DigiJazz's financial data, build and interpret regression models, "
     "identify trends and anomalies, and research relevant accounting standards or industry context. "
     "Be concise, precise, and frame all responses in an audit context. Cite sources where relevant."
     "Consider the auditor (with limited technical/statistical knowledge) and always tailor your responses to their needs."
     "In your greeting, briefly introduce yourself and your capabilities, and suggest how you can assist with the audit."),
    MessagesPlaceholder(variable_name="chat_history"),
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
            return_intermediate_steps=True, output_key="output",
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
    """Clears conversational buffer and all RAG documents."""
    memory.clear()
    _executor_cache.clear()

    # Delete all documents from the existing collection in-place — avoids
    # destroying and re-opening the chromadb file (which breaks the Rust client).
    ids = docs_vectorstore.get()["ids"]
    if ids:
        docs_vectorstore.delete(ids)

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

    # Re-populate the conversation buffer from saved history so the agent
    # has context when the app restarts. Only plain user/assistant text messages
    # are relevant — skip tool-call bubbles (they have metadata).
    for m in raw:
        if isinstance(m, dict) and not m.get("metadata"):
            content = m.get("content", "")
            if isinstance(content, str):  # skip image messages (content is a dict)
                if m["role"] == "user":
                    memory.chat_memory.add_user_message(content)
                elif m["role"] == "assistant":
                    memory.chat_memory.add_ai_message(content)

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
        Path(CHAT_LOG).unlink(missing_ok=True)
        cleared = [{"role": "assistant", "content": "🧹 Memory, PDF base, and chat history cleared."}]
        return cleared, cleared

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
        obs_str = str(observation)
        is_chart = obs_str.startswith("CHART_B64:")

        if is_chart:
            b64 = obs_str[len("CHART_B64:"):]
            data_url = f"data:image/png;base64,{b64}"
            bubble_content = f"**Input:** {tool_input}"
            chat_history.append(gr.ChatMessage(
                role="assistant",
                content=bubble_content,
                metadata={"title": f'The LLM used the tool "{action.tool}"', "status": "done"},
            ))
            # Render inline so Gradio shows the image regardless of file path sandboxing
            chat_history.append({"role": "assistant", "content": f"![DigiJazz chart]({data_url})"})
            # Inject as a vision message into memory so the LLM can see and describe it
            memory.chat_memory.add_message(HumanMessage(content=[
                {"type": "text", "text": "The chart has been generated and is now visible to the user."},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]))
        else:
            chat_history.append(gr.ChatMessage(
                role="assistant",
                content=f"**Input:** {tool_input}\n\n**Output:**\n```\n{obs_str}\n```",
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