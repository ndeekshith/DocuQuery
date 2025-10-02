"""
DocuQuery - Advanced Document Q&A System
Main Streamlit Application
"""

import os
# Force CPU usage before any imports
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TORCH_USE_CUDA_DSA'] = '0'

import streamlit as st
import asyncio
from pathlib import Path

from src.config import config
from src.logger import logger
from src.database import DocumentDatabase
from src.document_loaders import DocumentLoader
from src.vectorstore import VectorStoreManager
from src.qa_chain import QAChainManager
from src.ui_components import (
    apply_theme,
    render_statistics,
    render_document_list,
    render_source_documents,
    render_chat_message,
    toggle_dark_mode,
    show_progress_bar
)

# Page configuration
st.set_page_config(
    page_title=config.ui.page_title,
    page_icon=config.ui.page_icon,
    layout=config.ui.layout
)

# Apply custom theme
apply_theme()

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables."""
    if 'initialized' not in st.session_state:
        logger.info("Initializing application...")
        
        st.session_state.db = DocumentDatabase()
        st.session_state.doc_loader = DocumentLoader()
        st.session_state.vectorstore_manager = VectorStoreManager()
        st.session_state.qa_manager = QAChainManager(st.session_state.vectorstore_manager)
        
        st.session_state.conversation_history = []
        st.session_state.available_models = st.session_state.qa_manager.get_available_models()
        st.session_state.selected_model = st.session_state.available_models[0] if st.session_state.available_models else config.ollama.default_model
        st.session_state.system_prompt = st.session_state.qa_manager.system_prompt
        
        # Initialize QA chain if vectorstore exists
        if st.session_state.vectorstore_manager.vectorstore is not None:
            st.session_state.qa_manager.initialize_chain(
                st.session_state.selected_model,
                st.session_state.system_prompt
            )
        
        st.session_state.initialized = True
        logger.info("Application initialized successfully")

initialize_session_state()

# Header
st.title("📄 DocuQuery - Advanced Document Q&A")
st.markdown("*Powered by Local LLMs, FAISS Vector Search, and Hybrid Retrieval*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Dark mode toggle
    if config.ui.enable_dark_mode:
        if st.button("🌓 Toggle Dark Mode"):
            toggle_dark_mode()
    
    st.divider()
    
    # Model selection
    st.subheader("🧠 LLM Settings")
    selected_model = st.selectbox(
        "Select Ollama Model:",
        options=st.session_state.available_models,
        index=st.session_state.available_models.index(st.session_state.selected_model) if st.session_state.selected_model in st.session_state.available_models else 0,
        key="model_selector"
    )
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt:",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_editor"
    )
    
    if st.button("💾 Update Settings", key="update_settings"):
        st.session_state.selected_model = selected_model
        st.session_state.system_prompt = system_prompt
        
        if st.session_state.vectorstore_manager.vectorstore is not None:
            with st.spinner("Updating QA chain..."):
                success = st.session_state.qa_manager.initialize_chain(
                    selected_model,
                    system_prompt,
                    use_streaming=config.ui.streaming_enabled
                )
            if success:
                st.success("✅ Settings updated!")
            else:
                st.error("❌ Failed to update settings")
        else:
            st.warning("⚠️ Please upload documents first")
    
    st.divider()
    
    # Document upload
    st.subheader("📤 Upload Documents")
    
    supported_formats = config.document_processing.supported_formats
    uploaded_files = st.file_uploader(
        f"Upload files ({', '.join(supported_formats).upper()})",
        type=supported_formats,
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    if st.button("🚀 Process Documents", key="process_button", disabled=not uploaded_files):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                progress_bar = st.progress(0, text="Starting...")
                
                # Load documents with progress tracking
                all_chunks = []
                total_files = len(uploaded_files)
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    progress_bar.progress(
                        (idx + 1) / total_files,
                        text=f"Processing {uploaded_file.name} ({idx + 1}/{total_files})"
                    )
                    
                    # Save and process file
                    temp_dir = Path(config.document_processing.temp_directory)
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    file_path = temp_dir / uploaded_file.name
                    
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    chunks, doc_id = st.session_state.doc_loader.load_single_document(
                        str(file_path),
                        uploaded_file.name
                    )
                    all_chunks.extend(chunks)
                    
                    # Cleanup
                    try:
                        os.remove(file_path)
                    except:
                        pass
                
                # Create/update vectorstore
                if all_chunks:
                    progress_bar.progress(1.0, text="Creating embeddings...")
                    success = st.session_state.vectorstore_manager.create_vectorstore(all_chunks)
                    
                    if success:
                        # Initialize QA chain
                        st.session_state.qa_manager.initialize_chain(
                            st.session_state.selected_model,
                            st.session_state.system_prompt,
                            use_streaming=config.ui.streaming_enabled
                        )
                        st.success(f"✅ Processed {total_files} document(s) with {len(all_chunks)} chunks!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to create vector store")
                else:
                    st.warning("⚠️ No content extracted from documents")
    
    st.divider()
    
    # Statistics
    if st.session_state.vectorstore_manager.vectorstore is not None:
        st.subheader("📊 Quick Stats")
        db_stats = st.session_state.db.get_statistics()
        vs_stats = st.session_state.vectorstore_manager.get_statistics()
        
        st.metric("Documents", db_stats.get('total_documents', 0))
        st.metric("Chunks", db_stats.get('total_chunks', 0))
        st.metric("Vectors", vs_stats.get('total_vectors', 0))
        
        # Clear button with confirmation flow
        if st.button("🗑️ Clear All Data", key="clear_all"):
            st.session_state["confirm_clear_all"] = True
        
        if st.session_state.get("confirm_clear_all"):
            st.warning("Are you sure? This will delete all documents and embeddings.")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("✅ Confirm", key="confirm_clear_yes"):
                    st.session_state.vectorstore_manager.clear_vectorstore()
                    st.session_state.db.clear_all_documents()
                    st.session_state.conversation_history = []
                    st.session_state.qa_manager.qa_chain = None
                    st.session_state["confirm_clear_all"] = False
                    st.success("✅ All data cleared!")
                    st.rerun()
            with col_c2:
                if st.button("❌ Cancel", key="confirm_clear_no"):
                    st.session_state["confirm_clear_all"] = False

# Main content area
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Statistics", "📚 Documents"])

# Tab 1: Chat Interface
with tab1:
    st.subheader("Ask Questions")
    
    if st.session_state.vectorstore_manager.vectorstore is None:
        st.info("👈 Please upload and process documents using the sidebar to get started.")
    else:
        # Display chat history
        if st.session_state.conversation_history:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.conversation_history = []
                st.rerun()
            
            for message in st.session_state.conversation_history:
                render_chat_message(message, message.get('query', ''))
        
        # Chat input
        user_question = st.chat_input("Ask a question about your documents...", key="chat_input")
        
        if user_question:
            # Add user message
            st.session_state.conversation_history.append({
                "type": "user",
                "content": user_question,
                "query": user_question
            })
            
            # Display user message
            with st.chat_message("user", avatar="❓"):
                st.write(user_question)
            
            # Get response
            with st.chat_message("assistant", avatar="🤖"):
                if config.ui.streaming_enabled:
                    response_container = st.empty()
                    response = st.session_state.qa_manager.query(
                        user_question,
                        streaming_container=response_container
                    )
                else:
                    with st.spinner("Thinking..."):
                        response = st.session_state.qa_manager.query(user_question)
                    st.write(response.get('result', 'No answer generated.'))
                
                # Display sources
                source_docs = response.get('source_documents', [])
                if source_docs:
                    render_source_documents(source_docs, user_question)
            
            # Add to history
            st.session_state.conversation_history.append({
                "type": "assistant",
                "content": response.get('result', 'No answer generated.'),
                "sources": source_docs,
                "query": user_question
            })

# Tab 2: Statistics
with tab2:
    if st.session_state.vectorstore_manager.vectorstore is not None:
        db_stats = st.session_state.db.get_statistics()
        vs_stats = st.session_state.vectorstore_manager.get_statistics()
        
        combined_stats = {
            **db_stats,
            **vs_stats
        }
        
        render_statistics(combined_stats)
        
        # Additional info
        st.divider()
        st.subheader("🔧 System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Embedding Model:** {config.embeddings.model_name}")
            st.write(f"**LLM Model:** {st.session_state.selected_model}")
            st.write(f"**Vector Store:** {config.vectorstore.type.upper()}")
        
        with col2:
            st.write(f"**Hybrid Search:** {'✅ Enabled' if config.search.hybrid_search else '❌ Disabled'}")
            st.write(f"**Reranking:** {'✅ Enabled' if config.search.use_reranker else '❌ Disabled'}")
            st.write(f"**Streaming:** {'✅ Enabled' if config.ui.streaming_enabled else '❌ Disabled'}")
    else:
        st.info("No data available. Please upload documents first.")

# Tab 3: Document Management
with tab3:
    documents = st.session_state.db.get_all_documents()
    
    def delete_document(doc_id):
        """Delete a document."""
        st.session_state.db.delete_document(doc_id)
        st.success(f"Document {doc_id} deleted!")
        # Note: This doesn't remove from vectorstore - would need rebuild
        st.info("Note: Vectorstore rebuild required to fully remove embeddings.")
        st.rerun()
    
    def reprocess_document(doc_id):
        """Reprocess a failed document."""
        st.info(f"Reprocessing document {doc_id}...")
        # Implementation would require storing original file
        st.warning("Reprocessing requires re-uploading the file.")
    
    render_document_list(documents, delete_document, reprocess_document)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>DocuQuery v2.0 | Built by Deekshith N</p>
    <p>All processing happens locally - your data never leaves your machine</p>
</div>
""", unsafe_allow_html=True)

logger.info("App render complete")
