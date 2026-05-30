# 📚 WikiRAG — Knowledge Base Chatbot

WikiRAG is an interactive chatbot that lets you enter **any** Wikipedia topic (e.g. "Black holes"), automatically scrapes the full article text, indexes it into a local, in-memory **FAISS vector database**, and hosts a multi-turn conversation with memory using **Ollama** locally on your machine—100% free and private!

## ✨ Key Features
- **🌐 Dynamic Wikipedia Scraper**: Enter any topic to download and build a localized knowledge base on-the-fly.
- **⚡ In-Memory Vector Search**: Uses the recursive character text splitter and **FAISS** to index and retrieve relevant snippets instantly.
- **💬 Conversational Memory**: Uses LangChain's `ConversationBufferMemory` to maintain context over multi-turn chats.
- **🦙 Ollama Local Integration**: Out-of-the-box integration with Ollama's local embedding (`nomic-embed-text`) and chat model (`llama3`).
- **🛡️ Auto-Detect & Fallback Engine**: If Ollama is not running on your machine, the system automatically falls back to an intelligent, built-in **Local Simulation Engine** (uses a custom in-memory word frequency vectorizer and deterministic QA engine) to let you test the interface immediately without any setup or API keys!
- **🎨 Premium Gradio UI**: Sleek, modern chat interface designed with Gradio Blocks.

---

## 🚀 Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.10+ installed.

Clone the repository and install the dependencies:
```bash
git clone https://github.com/yashraj4/WikiRAG.git
cd WikiRAG

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install required packages
pip install wikipedia-api langchain langchain-community faiss-cpu gradio requests numpy tiktoken langchain-openai
```

### 2. Setting up Ollama (For Free Local AI)
To chat with a real offline LLM on your hardware:
1. Download and run the **[Ollama Desktop App](https://ollama.com/)**.
2. Download the default model weights by running the following in your terminal:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```

### 3. Running the App
Run the Gradio web interface locally:
```bash
python app.py
```
Open **[http://127.0.0.1:7861](http://127.0.0.1:7861)** in your browser!

---

## 🛠️ How it Works under the Hood

1. **Scraping**: Fetches structural text using `wikipedia-api` dynamically.
2. **Chunking**: Splits the long Wikipedia page into smaller, cohesive blocks of 1,000 characters (200 overlap) using LangChain's `RecursiveCharacterTextSplitter`.
3. **Indexing**: Embeds chunks and inserts them into a local `FAISS` vector store in memory.
4. **Retrieval**: Performs a similarity search on the FAISS index using the user's chat input to retrieve the top `2` most relevant chunks.
5. **Memory**: Links the retrieved text with historical user queries inside `ConversationBufferMemory`.
6. **Inference**: Passes the final prompt context containing memory + raw chunks to Ollama (`llama3`) to construct a highly factual, hallucination-free response.

---

## 📝 License
Distributed under the MIT License. See `LICENSE` for more information.
