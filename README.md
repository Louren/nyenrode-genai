<img src="logo_nyenrode.svg" height="28">

# Agentic AI — DigiJazz Audit Assistant

An AI audit assistant with persistent memory, regression analysis, and retrieval-augmented generation (RAG), built with LangChain and Gradio. Tailored for auditors working on the DigiJazz.nl engagement.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/louren/nyenrode-genai/blob/main/agent.ipynb)


## Features

- **Audit-focused assistant** — framed for the DigiJazz audit context (streaming/vinyl platform, founded 2015)
- **DigiJazz regression tools** — explore the dataset, list cost features, train OLS models, and evaluate out-of-sample performance
- **DigiJazz chart** — visualize weekly revenue and all cost drivers in a single chart
- Web search via DuckDuckGo
- Wikipedia and Arxiv search
- PDF upload and RAG (retrieval-augmented generation)
- Persistent conversational memory via ChromaDB
- Persistent chat log saved to JSON (survives restarts)
- Visible thinking steps — see tool calls and model reasoning in the chat
- Model picker — switch between GPT models at runtime
- Gradio chat UI with public share link

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key to a `.env` file:

```
OPENAI_API_KEY=sk-...
```

Then run:

```bash
python agent.py
```

## Usage

- Ask any question — the agent picks the right tool automatically.
- Ask about the DigiJazz dataset, cost features, or train a regression model.
- Upload PDFs using the file picker, then ask questions about them.
- Switch between models using the dropdown in the chat UI.
- Type `/clear` to reset memory, the PDF database, and the chat log.
