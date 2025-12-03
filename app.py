import streamlit as st
import tempfile
import os
from rag_core import create_vector_store, build_rag_graph

# --- Streamlit UI Setup ---
st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    layout="wide",
)

st.title("🤖 LangGraph RAG Chatbot (Ollama + FAISS)")
st.caption("Orchestration via LangGraph. Powered by Deepseek-R1 & nomic-embed-text.")

# --- Session State Initialization ---
if "rag_graph" not in st.session_state:
    st.session_state["rag_graph"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Welcome! Please upload a PDF document on the sidebar to begin our conversation."}
    ]

# --- Sidebar for Document Processing ---
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type="pdf",
        accept_multiple_files=False,
        help="The document will be processed and used as context for the chatbot."
    )

    if st.button("Process Document & Load Chatbot", use_container_width=True):
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing document and building LangGraph RAG system..."):
                try:
                    # 1. Create vector store (Ingestion step)
                    create_vector_store(tmp_file_path)
                    
                    # 2. Build the LangGraph application
                    st.session_state["rag_graph"] = build_rag_graph()
                    
                    st.session_state["messages"] = [
                        {"role": "assistant", "content": f"Document '{uploaded_file.name}' processed! LangGraph is ready. Ask me anything about it."}
                    ]
                    st.success("LangGraph RAG system is ready!")
                except Exception as e:
                    st.error(f"An error occurred during document processing: {e}")
                finally:
                    os.remove(tmp_file_path)
        else:
            st.warning("Please upload a PDF file first.")

    # Option to load a previously created index
    st.markdown("---")
    if st.button("Load Chatbot from Existing Index", use_container_width=True):
        with st.spinner("Initializing LangGraph from existing FAISS index..."):
            try:
                # We only need to rebuild the graph, the retriever inside it will load the index
                st.session_state["rag_graph"] = build_rag_graph()
                st.session_state["messages"] = [
                    {"role": "assistant", "content": "LangGraph RAG system loaded from existing FAISS index! Ask away."}
                ]
                st.success("Chatbot loaded successfully!")
            except FileNotFoundError:
                st.error("No existing FAISS index found. Please process a document first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Main Chat Interface ---

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
    if st.session_state["rag_graph"] is None:
        st.warning("Please process or load a document first in the sidebar.")
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
    else:
        # 1. Display user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Get assistant response using LangGraph's invoke
        with st.chat_message("assistant"):
            with st.spinner("LangGraph is running the RAG workflow (Retrieve -> Generate)..."):
                try:
                    # LangGraph requires the initial state as input
                    initial_state = {"question": prompt, "documents": [], "answer": ""}
                    
                    # Invoke the compiled graph
                    final_state = st.session_state["rag_graph"].invoke(initial_state)
                    
                    # The final answer is stored in the 'answer' field of the final state
                    answer = final_state["answer"]
                    st.markdown(answer)
                    
                    # 3. Add assistant response to history
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_message = f"Error generating response. Ensure Ollama is running and models are pulled. Error: {e}"
                    st.error(error_message)
                    st.session_state["messages"].append({"role": "assistant", "content": error_message})


# --- How to Run ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 🚀 Installation Check
    1. **Install Dependencies (Required New Packages):** `pip install streamlit langgraph langchain-community faiss-cpu pypdf langchain-text-splitters`
    2. **Start Ollama:** `ollama serve` (in a separate terminal)
    3. **Pull Models:** `ollama pull deepseek-r1:1.5b` and `ollama pull nomic-embed-text:latest`
    4. **Run App:** `streamlit run app.py`
    """
)