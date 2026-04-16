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

PERSIST_BASE = Path("/content/agentic_ai_persist")
PERSIST_BASE.mkdir(parents=True, exist_ok=True)

MEMORY_DIR = str(PERSIST_BASE / "chroma_memory")        # conversational memory
DOCS_DIR   = str(PERSIST_BASE / "chroma_docs")          # PDF database
UPLOAD_DIR = Path("/content/uploads")
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
pubmed = PubMedAPIWrapper()

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
        name="PubMed_Search",
        func=pubmed.run,
        description="Finds biomedical articles on PubMed. Input: query (e.g. 'depression CBT RCT')."
    ),
    Tool(
        name="RAG_PDFs",
        func=rag_search,
        description="Searches your uploaded PDFs (RAG). Input: question or term."
    ),
]

# ============================== Model & Prompt ============================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AI assistant with access to tools (web, wikipedia, arxiv, pubmed, RAG) "
     "and persistent memory. Cite sources when possible. Be concise and helpful."),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, memory=memory)

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
- **Academic:** Ask for academic topics — the agent may use Arxiv or PubMed automatically.
- **Your PDFs (RAG):** Upload files below, then ask questions about them.
- **/clear:** Erases all memory and your PDF database.
"""

def respond(msg, chat_history):
    # Command to clear memory
    if msg.strip().lower() == "/clear":
        clear_all_memory()
        chat_history = chat_history or []
        chat_history.append(("/clear", "🧹 Memory and PDF base cleared successfully."))
        return chat_history, chat_history

    try:
        response = executor.invoke({"input": msg})["output"]
    except Exception as e:
        response = f"⚠️ Error: {e}"

    chat_history = chat_history or []
    chat_history.append((msg, response))
    return chat_history, chat_history

def upload_pdfs(files, chat_history):
    """Handles uploaded PDFs and indexes them."""
    if not files:
        chat_history = chat_history or []
        chat_history.append(("Upload", "No files uploaded."))
        return chat_history, chat_history

    saved_paths = []
    for f in files:
        dest = UPLOAD_DIR / f.name
        with open(dest, "wb") as w:
            w.write(f.read())
        saved_paths.append(str(dest))

    chunks = index_pdfs(saved_paths)
    chat_history = chat_history or []
    chat_history.append(("Upload",
                         f"📚 {len(files)} file(s) uploaded. "
                         f"Indexed {chunks} text chunks into RAG. You can now ask questions!"))
    return chat_history, chat_history

with gr.Blocks(title="Agentic AI — Web + Wikipedia + Arxiv + PubMed + RAG PDFs") as demo:
    gr.Markdown("# 🤖 Agentic AI with Persistent Memory and RAG (PDFs)")
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