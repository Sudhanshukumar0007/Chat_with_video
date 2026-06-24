# 🎥 Chat with YouTube Video – RAG App

An interactive, AI-powered Streamlit web application that allows users to paste any YouTube video URL, automatically extract/process its transcript, and chat with the content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to ensure answers are grounded **only** in the video's transcript.

---

## 🚀 Features

- **YouTube Transcript Extraction:** Automatically extracts transcripts from standard URLs, shortened links (`youtu.be`), and embedded video links.
- **RAG Architecture:** Employs semantic search to retrieve the most relevant transcript segments.
- **Strict Grounding:** The assistant answers queries based *solely* on the transcript context. If the transcript doesn't contain the answer, it transparently states so.
- **Streamlit Chat Interface:** A clean, side-by-side user interface featuring setup configuration, processing status indicators, and an interactive chat window.
- **Reset Capabilities:** Easily clear current video state, chat history, and load new content.

---

## 🛠️ Technical Stack

- **Frontend:** [Streamlit](https://streamlit.io/) for rapid UI development.
- **Orchestration:** [LangChain](https://www.langchain.com/) (LangChain Core, Expression Language, Runnables).
- **Transcript Source:** `youtube-transcript-api` to pull video transcripts.
- **Text Chunking:** `RecursiveCharacterTextSplitter` (LangChain Classic).
- **Embeddings:** `HuggingFaceEmbeddings` using `sentence-transformers/all-MiniLM-L6-v2`.
- **Vector Database:** `FAISS` (Facebook AI Similarity Search) for local vector storage and similarity search.
- **Language Model:** `meta-llama/Llama-3.1-8B-Instruct` served via Hugging Face Hub (`HuggingFaceEndpoint` and `ChatHuggingFace`).

---

## 📂 Project Structure

```text
yt_chat_app/
│
├── app.py                  # Main Streamlit web application & user interface
├── rag.py                  # RAG pipeline implementation (transcript extraction, chunking, indexing, and LLM querying)
├── .gitignore              # Git ignore rules for virtual envs, secrets, and caches
└── README.md               # Project documentation
```

---

## 📦 Setup & Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd yt_chat_app
```

### 2. Set Up Virtual Environment
```bash
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages:
```bash
pip install streamlit youtube-transcript-api langchain-classic langchain-huggingface langchain-community faiss-cpu python-dotenv
```
*(Note: If you run into issues installing `faiss-cpu` on certain systems, check compatibility or use precompiled binaries.)*

### 4. Configure Secrets
Create a `.env` file in the root directory:
```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
```
Get your API token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

---

## 🎮 Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

1. Open your browser to `http://localhost:8501`.
2. Enter a YouTube video URL in the sidebar (e.g., `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
3. Click **Process Video**.
4. Once processing is complete, ask questions about the video in the main chat input field!