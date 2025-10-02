"""QA chain management with streaming support."""

import requests
from typing import Optional, List, Iterator
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.base import BaseCallbackHandler

from src.config import config
from src.logger import logger
from src.vectorstore import VectorStoreManager


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self, container):
        self.container = container
        self.text = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.text += token
        self.container.markdown(self.text)


class QAChainManager:
    """Manages QA chain with LLM integration."""
    
    def __init__(self, vectorstore_manager: VectorStoreManager):
        self.config = config
        self.vectorstore_manager = vectorstore_manager
        self.qa_chain = None
        self.current_model = None
        self.system_prompt = self._get_default_system_prompt()
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt."""
        return """You are a helpful AI assistant. Your task is to answer questions based ONLY on the provided context.
If the information is not in the context, say "I don't know" or "The provided context does not contain this information."
Do not make up information or use external knowledge. Be concise and accurate."""
    
    def get_available_models(self) -> List[str]:
        """Get available Ollama models."""
        try:
            response = requests.get(
                f"{config.ollama.base_url}/api/tags",
                timeout=config.ollama.timeout
            )
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                logger.info(f"Found {len(models)} Ollama models")
                return models
            else:
                logger.warning(f"Failed to fetch models: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
        
        # Fallback
        return [config.ollama.default_model]
    
    def initialize_chain(self, model_name: Optional[str] = None, 
                        system_prompt: Optional[str] = None,
                        use_streaming: bool = False) -> bool:
        """Initialize or update QA chain."""
        model_name = model_name or config.ollama.default_model
        system_prompt = system_prompt or self.system_prompt
        
        retriever = self.vectorstore_manager.get_retriever()
        if retriever is None:
            logger.error("Cannot initialize QA chain: retriever not available")
            return False
        
        try:
            logger.info(f"Initializing QA chain with model: {model_name}")
            
            # Create prompt template
            prompt_template = system_prompt + """

Context:
{context}

Question: {question}

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Initialize LLM
            llm_kwargs = {
                'model': model_name,
                'temperature': config.ollama.temperature,
                'base_url': config.ollama.base_url
            }
            
            if use_streaming and config.ui.streaming_enabled:
                llm = ChatOllama(**llm_kwargs, streaming=True)
            else:
                llm = ChatOllama(**llm_kwargs)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            
            self.current_model = model_name
            self.system_prompt = system_prompt
            
            logger.info("QA chain initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing QA chain: {e}")
            return False
    
    def query(self, question: str, streaming_container=None) -> dict:
        """Query the QA chain."""
        if self.qa_chain is None:
            logger.error("QA chain not initialized")
            return {
                'result': "QA system not ready. Please process documents first.",
                'source_documents': []
            }
        
        try:
            logger.info(f"Processing query: {question[:50]}...")
            
            # Add streaming callback if container provided
            callbacks = []
            if streaming_container and config.ui.streaming_enabled:
                callbacks.append(StreamingCallbackHandler(streaming_container))
            
            if callbacks:
                response = self.qa_chain.invoke(
                    {"query": question},
                    {"callbacks": callbacks}
                )
            else:
                response = self.qa_chain.invoke({"query": question})
            
            logger.info("Query processed successfully")
            return response
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'result': f"Error: {str(e)}",
                'source_documents': []
            }
    
    def highlight_text(self, text: str, query: str) -> str:
        """Highlight query terms in text."""
        if not config.ui.show_source_highlighting:
            return text
        
        # Simple highlighting - can be enhanced with better NLP
        words = query.lower().split()
        highlighted = text
        
        for word in words:
            if len(word) > 3:  # Only highlight words longer than 3 chars
                # Case-insensitive replacement with HTML highlighting
                import re
                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(
                    lambda m: f"**{m.group()}**",
                    highlighted
                )
        
        return highlighted
