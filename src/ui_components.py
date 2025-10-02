"""UI components and utilities for Streamlit app."""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime

from src.config import config
from src.logger import logger


def apply_theme():
    """Apply modern theme with smooth transitions and clean design."""
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = config.ui.default_theme == 'dark'
    
    is_dark = st.session_state.dark_mode
    
    # Modern color palette
    colors = {
        'dark': {
            'app_bg': '#0a0e14',
            'sidebar_bg': '#131920',
            'card_bg': '#1a2332',
            'source_bg': '#162030',
            'text': '#e6edf3',
            'subtext': '#8b949e',
            'accent': '#58a6ff',
            'accent_secondary': '#7ee787',
            'border': '#30363d',
            'highlight': '#ffd60a',
            'shadow': 'rgba(0, 0, 0, 0.4)',
        },
        'light': {
            'app_bg': '#ffffff',
            'sidebar_bg': '#f6f8fa',
            'card_bg': '#ffffff',
            'source_bg': '#f6f8fa',
            'text': '#1f2937',
            'subtext': '#6b7280',
            'accent': '#2563eb',
            'accent_secondary': '#059669',
            'border': '#e5e7eb',
            'highlight': '#fbbf24',
            'shadow': 'rgba(0, 0, 0, 0.1)',
        }
    }
    
    theme = colors['dark'] if is_dark else colors['light']
    
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Base styles */
    html, body, .stApp {{
        background: {theme['app_bg']};
        color: {theme['text']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    
    .stApp {{
        max-width: 100%;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: {theme['sidebar_bg']} !important;
        border-right: 1px solid {theme['border']};
    }}
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {theme['text']};
    }}
    
    /* Typography */
    .stApp, .stMarkdown, .stText, p, span, label {{
        color: {theme['text']};
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {theme['text']} !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }}
    
    /* Modern stat cards */
    .stat-card {{
        background: {theme['card_bg']};
        padding: 24px;
        border-radius: 16px;
        border: 1px solid {theme['border']};
        box-shadow: 0 4px 6px -1px {theme['shadow']}, 0 2px 4px -1px {theme['shadow']};
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }}
    
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {theme['accent']}, {theme['accent_secondary']});
        opacity: 0;
        transition: opacity 0.3s ease;
    }}
    
    .stat-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -2px {theme['shadow']}, 0 4px 8px -2px {theme['shadow']};
    }}
    
    .stat-card:hover::before {{
        opacity: 1;
    }}
    
    .stat-number {{
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(135deg, {theme['accent']}, {theme['accent_secondary']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}
    
    .stat-label {{
        font-size: 13px;
        font-weight: 500;
        color: {theme['subtext']};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }}
    
    /* Source boxes */
    .source-box {{
        background: {theme['source_bg']};
        border: 1px solid {theme['border']};
        border-left: 4px solid {theme['accent']};
        padding: 20px;
        margin: 16px 0;
        border-radius: 12px;
        color: {theme['text']};
        white-space: pre-wrap;
        line-height: 1.8;
        overflow-x: auto;
        font-size: 14px;
        box-shadow: 0 2px 8px {theme['shadow']};
        transition: all 0.2s ease;
    }}
    
    .source-box:hover {{
        border-left-width: 6px;
        box-shadow: 0 4px 12px {theme['shadow']};
    }}
    
    .highlight {{
        background: {theme['highlight']};
        color: #1f2937;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 500;
    }}
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stTextArea textarea {{
        background: {theme['card_bg']} !important;
        color: {theme['text']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        transition: all 0.2s ease !important;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea textarea:focus {{
        border-color: {theme['accent']} !important;
        box-shadow: 0 0 0 3px {theme['accent']}22 !important;
        outline: none !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background: linear-gradient(135deg, {theme['accent']}, {theme['accent_secondary']}) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px -1px {theme['shadow']} !important;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 12px -2px {theme['shadow']} !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0);
    }}
    
    /* Delete and action buttons */
    button[kind="secondary"] {{
        background: {theme['card_bg']} !important;
        color: {theme['text']} !important;
        border: 1px solid {theme['border']} !important;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 2px solid {theme['border']};
        padding-bottom: 0;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {theme['subtext']};
        padding: 12px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: {theme['card_bg']};
        color: {theme['text']};
    }}
    
    .stTabs [aria-selected="true"] {{
        color: {theme['accent']} !important;
        background: {theme['card_bg']};
        border-bottom: 2px solid {theme['accent']};
    }}
    
    /* Chat messages */
    .stChatMessage {{
        background: {theme['card_bg']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 16px !important;
        padding: 20px !important;
        margin: 12px 0 !important;
        box-shadow: 0 2px 8px {theme['shadow']} !important;
    }}
    
    /* Expanders */
    .streamlit-expanderHeader {{
        background: {theme['card_bg']} !important;
        color: {theme['text']} !important;
        border: 1px solid {theme['border']} !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    
    .streamlit-expanderHeader:hover {{
        background: {theme['source_bg']} !important;
        border-color: {theme['accent']} !important;
    }}
    
    .streamlit-expanderContent {{
        background: {theme['app_bg']} !important;
        color: {theme['text']} !important;
        border: 1px solid {theme['border']} !important;
        border-top: none !important;
        border-radius: 0 0 12px 12px !important;
        padding: 20px !important;
    }}
    
    /* Metrics */
    .stMetric {{
        background: {theme['card_bg']};
        padding: 16px;
        border-radius: 10px;
        border: 1px solid {theme['border']};
    }}
    
    .stMetric label {{
        color: {theme['subtext']} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    
    .stMetric [data-testid="stMetricValue"] {{
        color: {theme['text']} !important;
        font-size: 24px !important;
        font-weight: 700 !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {theme['accent']}, {theme['accent_secondary']}) !important;
        border-radius: 10px !important;
    }}
    
    /* Info/Warning boxes */
    .stAlert {{
        border-radius: 12px !important;
        border: 1px solid {theme['border']} !important;
        background: {theme['card_bg']} !important;
    }}
    
    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: {theme['app_bg']};
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: {theme['border']};
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: {theme['subtext']};
    }}
    
    /* Smooth transitions */
    * {{
        transition: background-color 0.2s ease, border-color 0.2s ease, color 0.2s ease;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_statistics(stats: Dict[str, Any]):
    """Render modern statistics dashboard."""
    st.subheader("📊 Knowledge Base Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('total_documents', 0)}</div>
            <div class="stat-label">Documents</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats.get('total_chunks', 0)}</div>
            <div class="stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        size_mb = stats.get('total_size_bytes', 0) / (1024 * 1024)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{size_mb:.1f}</div>
            <div class="stat-label">MB Storage</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        vectors = stats.get('total_vectors', 0)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{vectors}</div>
            <div class="stat-label">Embeddings</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Document type breakdown
    if stats.get('documents_by_type'):
        st.markdown("### 📁 Document Types")
        type_data = stats['documents_by_type']
        cols = st.columns(len(type_data))
        for idx, (file_type, count) in enumerate(type_data.items()):
            with cols[idx]:
                st.metric(file_type.upper(), count)


def render_document_list(documents: List[Dict[str, Any]], on_delete_callback, on_reprocess_callback):
    """Render modern document management interface."""
    st.subheader("📚 Document Library")
    
    if not documents:
        st.info("📭 No documents yet. Upload your first document to get started!")
        return
    
    for doc in documents:
        status_emoji = "✅" if doc['status'] == 'processed' else "❌" if doc['status'] == 'failed' else "⏳"
        
        with st.expander(f"{status_emoji} {doc['filename']} · {doc['file_type'].upper()}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"""
                **Status:** `{doc['status'].title()}`  
                **Uploaded:** {format_datetime(doc['upload_date'])}  
                **Size:** {format_bytes(doc['file_size'])}  
                **Chunks:** {doc['chunk_count']}
                """)
                
                if doc.get('preview'):
                    st.markdown("**Preview:**")
                    preview_text = doc['preview'][:300] + "..." if len(doc['preview']) > 300 else doc['preview']
                    st.code(preview_text, language=None)
            
            with col2:
                if st.button("🗑️ Delete", key=f"delete_{doc['id']}", use_container_width=True):
                    on_delete_callback(doc['id'])
                
                if doc['status'] == 'failed':
                    if st.button("🔄 Retry", key=f"reprocess_{doc['id']}", use_container_width=True):
                        on_reprocess_callback(doc['id'])


def render_source_documents(source_docs: List[Any], query: str = ""):
    """Render source documents with modern styling."""
    if not source_docs:
        return
    
    st.markdown("### 📖 Sources")
    
    for idx, doc in enumerate(source_docs):
        source_file = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        with st.expander(f"📄 {source_file} (Page {page})", expanded=idx == 0):
            content = doc.page_content
            
            if config.ui.show_source_highlighting and query:
                content = highlight_query_terms(content, query)
            
            st.markdown(f'<div class="source-box">{content}</div>', unsafe_allow_html=True)


def highlight_query_terms(text: str, query: str) -> str:
    """Highlight query terms in text."""
    import re
    words = query.lower().split()
    highlighted = text
    
    for word in words:
        if len(word) > 3:
            pattern = re.compile(f'({re.escape(word)})', re.IGNORECASE)
            highlighted = pattern.sub(r'<span class="highlight">\1</span>', highlighted)
    
    return highlighted


def format_datetime(dt_string: str) -> str:
    """Format datetime for display."""
    try:
        dt = datetime.fromisoformat(dt_string)
        return dt.strftime("%b %d, %Y at %H:%M")
    except:
        return dt_string


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def show_progress_bar(current: int, total: int, text: str = "Processing"):
    """Show modern progress indicator."""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{text} • {current} of {total}")


def render_chat_message(message: Dict[str, Any], query: str = ""):
    """Render a chat message with modern styling."""
    if message['type'] == 'user':
        with st.chat_message("user", avatar="💬"):
            st.write(message['content'])
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message['content'])
            
            if message.get('sources'):
                render_source_documents(message['sources'], query)


def toggle_dark_mode():
    """Toggle dark mode with smooth transition."""
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.rerun()