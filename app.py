import os
import re
import requests
import numpy as np
import gradio as gr
import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from typing import Any, List, Optional

# Custom Bag-of-Words Embedding Class for Fallback
class SimpleBOWEmbeddings(Embeddings):
    def __init__(self, texts: List[str]):
        vocab = set()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            vocab.update(words)
        self.vocab = sorted(list(vocab))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.dimension = max(128, len(self.vocab))
        
    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dimension)
        words = re.findall(r'\w+', text.lower())
        for word in words:
            if word in self.word_to_idx:
                vector[self.word_to_idx[word]] += 1
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
        
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        return [self._embed(d) for d in documents]
        
    def embed_query(self, query: str) -> List[float]:
        return self._embed(query)

# Custom Mock LLM Class for Fallback
class LocalMockLLM(LLM):
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        # Search for context chunks in different possible LangChain prompt formats
        context = ""
        context_patterns = [
            r"Context:\n(.*?)\n\n",
            r"context:\s*(.*?)\n\n",
            r"use the following pieces of context to answer the question.*?\n\n(.*?)\n\n",
            r"System:\s*Use the following pieces of context to answer the user's question\.\n-+\n(.*?)\n-+\n",
            r"system\nUse the following pieces of context to answer the user's question\.\n(.*?)\nhuman"
        ]
        
        for pattern in context_patterns:
            match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
            if match:
                context = match.group(1).strip()
                break
        
        if context:
            paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
            best_chunk = paragraphs[0] if paragraphs else context.strip()
            best_chunk = re.sub(r'\[\d+\]', '', best_chunk)
            
            return (
                f"🤖 **[Ollama Simulation Mode]**\n\n"
                f"Based on the retrieved Wikipedia knowledge:\n"
                f"> {best_chunk}\n\n"
                f"⚠️ *Note: Operating in Local Simulation because we could not connect to your local Ollama instance.* "
                f"To unlock real AI responses:\n"
                f"1. Start the **Ollama Desktop App** on your computer.\n"
                f"2. Pull the models by running these in your terminal:\n"
                f"   `ollama pull llama3`\n"
                f"   `ollama pull nomic-embed-text`"
            )
        else:
            return "🤖 **[Ollama Simulation Mode]** No Wikipedia context was found. Please search and build a knowledge base first!"
            
    @property
    def _llm_type(self) -> str:
        return "local_mock"

# Check if Ollama is available
is_ollama_available = False
embeddings = None
llm = None
ollama_error_msg = ""

try:
    # Check if Ollama server is running
    response = requests.get("http://localhost:11434", timeout=2)
    if response.status_code == 200:
        # Attempt to load llama3 and nomic-embed-text
        llm = Ollama(model="llama3")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Test if the model is ready and pulled by doing a tiny query
        embeddings.embed_query("test")
        is_ollama_available = True
        print("Ollama server and models detected successfully!")
except Exception as e:
    ollama_error_msg = str(e)
    print(f"Ollama verification failed ({e}). Falling back to Local Simulation.")
    is_ollama_available = False

# Global variables to store state
vector_store = None
conversation_chain = None

def load_wikipedia_topic(topic):
    global vector_store, conversation_chain
    if not topic.strip():
        return "Please enter a topic."
    
    # Init Wikipedia API
    wiki_wiki = wikipediaapi.Wikipedia('WikiRAG/1.0 (test@example.com)', 'en')
    page = wiki_wiki.page(topic)
    
    if not page.exists():
        return f"Could not find Wikipedia page for '{topic}'."
    
    text = page.text
    if not text:
        return f"No content found for '{topic}'."
        
    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Convert to documents
    docs = [Document(page_content=t) for t in chunks]
    
    # Create Vector Store and LLM Chain depending on availability
    if is_ollama_available:
        vector_store = FAISS.from_documents(docs, embeddings)
        active_llm = llm
        engine_name = "Ollama (llama3 + nomic-embed-text)"
    else:
        # Use our pure local BOW embedding and mock LLM
        local_embeddings = SimpleBOWEmbeddings(chunks)
        vector_store = FAISS.from_documents(docs, local_embeddings)
        active_llm = LocalMockLLM()
        engine_name = "Local Simulation Engine"
        
    # Create conversation chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=active_llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )
    
    status = f"Knowledge base built for '{topic}' using {engine_name}! You can now chat below."
    if not is_ollama_available:
        status += "\n\n(Ollama app not running, or 'llama3' / 'nomic-embed-text' models have not been pulled yet)."
    return status

def chat(message, history):
    global conversation_chain
    if conversation_chain is None:
        return "Please load a Wikipedia topic first."
    
    response = conversation_chain.invoke({"question": message})
    return response["answer"]

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📚 WikiRAG — Ollama Knowledge Base Chatbot")
    gr.Markdown("Type any topic, and the bot will scrape Wikipedia, build a local knowledge base, and let you chat about it using **Ollama**.")
    
    with gr.Row():
        with gr.Column(scale=4):
            topic_input = gr.Textbox(label="Wikipedia Topic", placeholder="e.g. Black holes", show_label=False)
        with gr.Column(scale=1):
            load_btn = gr.Button("Build Knowledge Base", variant="primary")
        
    status_output = gr.Textbox(label="Status", interactive=False)
    
    load_btn.click(fn=load_wikipedia_topic, inputs=topic_input, outputs=status_output)
    
    gr.Markdown("### Chat Interface")
    chatbot = gr.ChatInterface(
        fn=chat
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861, theme=gr.themes.Soft())
