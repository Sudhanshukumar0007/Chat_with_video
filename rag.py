from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import os

# -------------------- VIDEO ID EXTRACTION --------------------
def extract_video_id(url: str) -> str:
    parsed_url = urlparse(url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        if "v" in query_params:
            return query_params["v"][0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    if "embed" in parsed_url.path:
        return parsed_url.path.split("/")[-1]

    raise ValueError("Invalid YouTube URL")


# -------------------- PIPELINE BUILDING --------------------
def process_video(video_url):
    print("RAG received video url:", video_url)

    # 1. Extract ID
    video_id = extract_video_id(video_url)

    # 2. Fetch transcript
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)

    transcript = " ".join(snippet.text for snippet in fetched_transcript)

    # 3. Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([transcript])

    # 4. Embeddings + Vectorstore
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 5. Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 6. LLM (loaded once per video)
    load_dotenv()
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        huggingfacehub_api_token=hf_token
    )
    model = ChatHuggingFace(llm=llm)

    # 7. Prompt
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer only from the provided transcript context.
If context is not sufficient, say you don't know.

{context}

Question: {question}
""",
        input_variables=["context", "question"]
    )

    # 8. Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    return {
        "video_id": video_id,
        "chain": chain
    }


# -------------------- QUESTION ANSWERING --------------------
def ask_question(question, chain):
    print("RAG received question:", question)
    return chain.invoke(question)
