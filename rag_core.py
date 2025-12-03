import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- CORRECTED IMPORTS ---
# Core Dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
# -------------------------

# --- Configuration ---
LLM_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
VECTOR_DB_PATH = "faiss_index"

# --- 1. Define Graph State ---
# The state object passed between nodes in the graph
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's initial question (str)
        documents: List of retrieved documents (List[Document])
        answer: The final generated answer (str)
    """
    question: str
    documents: List[Document]
    answer: str

# --- 2. Setup Vector Store (Ingestion) ---

def create_vector_store(pdf_path: str) -> FAISS:
    """Loads PDF, splits it, generates embeddings, and creates a FAISS vector store."""
    print(f"Loading document from: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(chunks)} chunks.")

    print(f"Creating embeddings with model: {EMBEDDING_MODEL}")
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Creating FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"FAISS index saved to {VECTOR_DB_PATH}")
    
    return vector_store

def get_retriever():
    """Loads FAISS index and returns a retriever."""
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"FAISS index not found at '{VECTOR_DB_PATH}'. Please process a document first.")
    
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # Return the retriever instance
    return vector_store.as_retriever(search_kwargs={"k": 3})

# --- 3. Define Graph Nodes ---

def retrieve_node(state: GraphState):
    """
    Retrieves documents from the vector store based on the question.
    """
    print("---NODE: RETRIEVE DOCUMENTS---")
    question = state["question"]
    retriever = get_retriever()
    
    # LangChain retriever returns a list of Document objects
    documents = retriever.invoke(question)
    
    return {"documents": documents, "question": question}


def generate_node(state: GraphState):
    """
    Generates the final answer using the question and retrieved documents.
    """
    print("---NODE: GENERATE ANSWER---")
    question = state["question"]
    documents = state["documents"]
    
    # 1. Initialize Ollama LLM
    llm = Ollama(model=LLM_MODEL)
    
    # 2. Define the RAG Prompt Template
    template = (
        "You are an intelligent assistant. Use the following context to answer the user's question. "
        "If you don't know the answer, state that you don't know based on the provided context. "
        "Do not make up an answer."
        "\n\nContext: {context}"
        "\n\nQuestion: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    
    # 3. Format documents into a single string for the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    # 4. Create the LLM Chain (using LCEL)
    rag_chain = prompt | llm | StrOutputParser()

    # 5. Invoke the chain
    answer = rag_chain.invoke({"context": context_text, "question": question})
    
    return {"question": question, "documents": documents, "answer": answer}


# --- 4. Build the LangGraph Workflow ---

def build_rag_graph():
    """Constructs and compiles the LangGraph state machine for RAG."""
    
    # Initialize the StateGraph with the defined state object
    workflow = StateGraph(GraphState)

    # Add the nodes (steps)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Set the entry point (start of the graph)
    workflow.set_entry_point("retrieve")

    # Define the edges (flow of execution)
    workflow.add_edge("retrieve", "generate") # After retrieval, go to generation
    workflow.add_edge("generate", END)        # After generation, end the graph

    # Compile the graph
    app = workflow.compile()
    
    print("LangGraph RAG workflow compiled successfully.")
    return app