import streamlit as st
import os
import shutil # For removing directory
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM # For the basic LLM
from langchain_ollama.chat_models import ChatOllama # For chat models
from langchain.chains import RetrievalQA # More advanced QA chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory # For chat history context (if using conversational chain)

# --- Configuration ---
# OLLAMA_MODEL will now be selected via UI
DEFAULT_OLLAMA_MODEL = "phi3:3.8b-mini-4k-instruct-q5_K_M"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TEMP_DIR = "temp_docs"
VECTORSTORE_DIR = "vectorstore_db" # Directory to save FAISS index

# --- Helper Functions ---

def get_available_ollama_models():
    try:
        import requests
        # Use the REST API instead of the Python client
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            data = response.json()
            # Extract model names from the response
            return [model['name'] for model in data.get('models', [])]
        else:
            st.warning(f"Failed to fetch models from Ollama API: HTTP {response.status_code}")
    except ImportError:
        st.warning("Requests library not installed. Run `pip install requests` for API access.")
    except Exception as e:
        st.error(f"Could not fetch Ollama models: {str(e)}. Falling back to default model list.")
    
    # Fallback if API call fails
    return [
        "phi3:3.8b-mini-4k-instruct-q5_K_M",  
        "gemma2:2b"                         
    ]

def load_documents(uploaded_files):
    documents = []
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        try:
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}. Skipping.")
                continue
            documents.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {e}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    return documents

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def get_vectorstore(text_chunks, persist_directory=VECTORSTORE_DIR):
    if not text_chunks:
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # Check if a persisted store exists and load it
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            st.info(f"Loading existing vector store from {persist_directory}...")
            vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization
            st.info(f"Adding {len(text_chunks)} new chunks to existing store.")
            vectorstore.add_documents(text_chunks) # Add new chunks
        else:
            st.info("Creating new vector store...")
            vectorstore = FAISS.from_documents(text_chunks, embedding=embeddings)

        vectorstore.save_local(persist_directory) # Persist the (new or updated) vector store
        st.success(f"Vector store saved to {persist_directory}")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating/loading vector store: {e}")
        return None

def load_vectorstore_from_disk(persist_directory=VECTORSTORE_DIR):
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
            st.info(f"Loading vector store from {persist_directory}...")
            vectorstore = FAISS.load_local(persist_directory, embeddings, allow_dangerous_deserialization=True) # Added allow_dangerous_deserialization
            st.success("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            st.error(f"Error loading vector store from disk: {e}. Please re-process documents.")
            # Clean up potentially corrupted store
            if os.path.exists(persist_directory):
                try:
                    shutil.rmtree(persist_directory)
                    st.info(f"Cleaned up potentially corrupted store at {persist_directory}.")
                except Exception as e_clean:
                    st.error(f"Could not clean up corrupted store: {e_clean}")
            return None
    return None


def get_qa_chain(llm_model_name, system_prompt_template, vectorstore, use_chat_model=True):
    """Initializes the RetrievalQA chain."""
    if vectorstore is None:
        st.error("Vector store not available. Cannot create QA chain.")
        return None

    prompt_template = system_prompt_template + """

    Context:
    {context}

    Question: {question}

    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    if use_chat_model:
        llm = ChatOllama(model=llm_model_name, temperature=0.2)
    else:
        llm = OllamaLLM(model=llm_model_name, temperature=0.2)

    # Using RetrievalQA chain which is better for RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all context into the prompt
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}), # Retrieve top 5 chunks
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True # This is important to get source documents
    )
    return qa_chain

# --- Streamlit App ---
st.set_page_config(page_title="Advanced Doc Q&A", layout="wide")
st.header("📄 Advanced Document Q&A with Local LLM 💬")

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore_from_disk() # Try to load at startup
if "processed_files" not in st.session_state:
    st.session_state.processed_files = [] # This will be simple list, actual persistence is in vectorstore
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None # Will be initialized after model selection and vectorstore
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "available_ollama_models" not in st.session_state:
    st.session_state.available_ollama_models = get_available_ollama_models()
if "selected_ollama_model" not in st.session_state:
    st.session_state.selected_ollama_model = st.session_state.available_ollama_models[0] if st.session_state.available_ollama_models else DEFAULT_OLLAMA_MODEL
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """You are a helpful AI assistant. Your task is to answer questions based ONLY on the provided context.
If the information is not in the context, say "I don't know" or "The provided context does not contain this information."
Do not make up information or use external knowledge. Be concise and accurate."""
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# --- Sidebar ---
with st.sidebar:
    st.subheader("LLM Configuration")
    st.session_state.selected_ollama_model = st.selectbox(
        "Select Ollama Model:",
        options=st.session_state.available_ollama_models,
        index=st.session_state.available_ollama_models.index(st.session_state.selected_ollama_model) if st.session_state.selected_ollama_model in st.session_state.available_ollama_models else 0,
        key="ollama_model_selector"
    )

    st.subheader("System Prompt")
    st.session_state.system_prompt = st.text_area(
        "Customize the LLM's System Prompt:",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_editor"
    )
    if st.button("Update LLM & Prompt Settings", key="update_llm_prompt_btn"):
        if st.session_state.vectorstore:
            with st.spinner("Re-initializing QA chain with new settings..."):
                st.session_state.qa_chain = get_qa_chain(
                    st.session_state.selected_ollama_model,
                    st.session_state.system_prompt,
                    st.session_state.vectorstore
                )
            st.success("QA chain updated!")
        else:
            st.warning("Please process documents first to initialize the vector store.")


    st.divider()
    st.subheader("Upload & Process Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader_adv"
    )

    if st.button("Process Uploaded Documents", key="process_button_adv"):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                # For simplicity, we identify "new" files by name, but real tracking would be more robust
                # Current vector store handles adding new docs, so we don't need complex diffing here.
                raw_documents = load_documents(uploaded_files)
                if raw_documents:
                    text_chunks = get_text_chunks(raw_documents)
                    st.session_state.total_chunks += len(text_chunks) # Update total chunks
                    if text_chunks:
                        # get_vectorstore now handles loading existing and adding, or creating new
                        # It also saves to disk.
                        vs = get_vectorstore(text_chunks, persist_directory=VECTORSTORE_DIR)
                        if vs:
                            st.session_state.vectorstore = vs
                            # Re-initialize QA chain with potentially new vector store data
                            st.session_state.qa_chain = get_qa_chain(
                                st.session_state.selected_ollama_model,
                                st.session_state.system_prompt,
                                st.session_state.vectorstore
                            )
                            st.success(f"Processed {len(uploaded_files)} document(s). Total chunks: {st.session_state.total_chunks}. Knowledge base updated!")
                            st.session_state.processed_files.extend([f.name for f in uploaded_files]) # Simple tracking
                        else:
                            st.error("Failed to create or update vector store.")
                    else:
                        st.warning("No text chunks could be extracted from the documents.")
                else:
                    st.warning("No documents were successfully loaded.")
        else:
            st.warning("Please upload at least one document.")

    if st.session_state.vectorstore:
        st.success(f"Knowledge base active. Total Chunks: {st.session_state.total_chunks}")
        if st.button("⚠️ Clear Knowledge Base & History", key="clear_kb_button_adv"):
            with st.spinner("Clearing knowledge base..."):
                st.session_state.vectorstore = None
                st.session_state.processed_files = []
                st.session_state.conversation_history = []
                st.session_state.qa_chain = None
                st.session_state.total_chunks = 0
                if os.path.exists(VECTORSTORE_DIR):
                    try:
                        shutil.rmtree(VECTORSTORE_DIR) # Remove persisted vector store
                        st.success("Knowledge base directory cleared.")
                    except Exception as e:
                        st.error(f"Error clearing knowledge base directory: {e}")
            st.rerun()

# Initialize QA chain if vectorstore exists but chain doesn't (e.g., after loading from disk)
if st.session_state.vectorstore and not st.session_state.qa_chain:
    st.session_state.qa_chain = get_qa_chain(
        st.session_state.selected_ollama_model,
        st.session_state.system_prompt,
        st.session_state.vectorstore
    )
    if st.session_state.qa_chain:
        st.sidebar.info("QA chain initialized with loaded knowledge base.")
    else:
        st.sidebar.error("Failed to initialize QA chain with loaded knowledge base.")


# --- Main Q&A Area ---
st.subheader("Ask a Question")
st.markdown(f"Using Ollama model: `{st.session_state.selected_ollama_model}` and Embeddings: `{EMBEDDING_MODEL_NAME}`")

if st.session_state.conversation_history:
    if st.button("Clear Chat History", key="clear_chat_btn"):
        st.session_state.conversation_history = []
        st.rerun()

    for i, entry in enumerate(st.session_state.conversation_history):
        if entry["type"] == "user":
            st.chat_message("user", avatar="❓").write(entry["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(entry["content"])
                if "sources" in entry and entry["sources"]:
                    with st.expander("View Sources"):
                        for idx, doc in enumerate(entry["sources"]):
                            st.markdown(f"**Source {idx+1}:** (From `{doc.metadata.get('source', 'N/A')}`)")
                            st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)


user_question = st.chat_input("Your question:", key="user_question_input_adv")

if user_question:
    st.session_state.conversation_history.append({"type": "user", "content": user_question})
    st.chat_message("user", avatar="❓").write(user_question)

    if st.session_state.qa_chain is None:
        st.warning("QA system not ready. Please process documents or ensure LLM settings are correct.")
    else:
        with st.spinner("Searching for answer..."):
            try:
                response = st.session_state.qa_chain.invoke({"query": user_question})
                answer = response.get("result", "No answer generated.")
                source_documents = response.get("source_documents", [])

                assistant_response = {"type": "assistant", "content": answer, "sources": source_documents}
                st.session_state.conversation_history.append(assistant_response)

                with st.chat_message("assistant", avatar="🤖"):
                    st.write(answer)
                    if source_documents:
                        with st.expander("View Sources"):
                            for idx, doc in enumerate(source_documents):
                                st.markdown(f"**Source {idx+1}:** (From `{doc.metadata.get('source', 'N/A')}`)")
                                # Displaying a snippet of the source content
                                st.caption(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error(f"Ensure Ollama is running and the model '{st.session_state.selected_ollama_model}' is available.")
                st.session_state.conversation_history.append({"type": "assistant", "content": f"Error: {e}", "sources": []})

# Ensure temp_docs is removed if it's empty or at app shutdown (Streamlit doesn't have a clean shutdown hook)
# This is a simple cleanup; for robust temp file management, use `tempfile` module
if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
    os.rmdir(TEMP_DIR)