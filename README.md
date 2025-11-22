# Local Knowledge Chatbot

A local RAG (Retrieval-Augmented Generation) chatbot system that allows you to build a knowledge base from web URLs or text files and query it using a local LLM. This system runs entirely offline and uses Docker for easy deployment.

## Features

- 🔒 **Fully Local**: Runs completely offline with no external API dependencies
- 🌐 **Web Scraping**: Automatically scrapes and indexes content from web URLs
- 📄 **File Upload**: Supports uploading text files to expand the knowledge base
- 🧠 **RAG Architecture**: Uses Retrieval-Augmented Generation for accurate, context-aware responses
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 💾 **Persistent Storage**: Vector database persists across sessions
- 🎨 **Streamlit UI**: Clean, intuitive web interface

## Architecture

- **LLM**: Ollama (Mistral:instruct model)
- **Embeddings**: mxbai-embed-large (local Sentence Transformers model)
- **Vector Store**: ChromaDB
- **Framework**: LangChain
- **UI**: Streamlit

## Prerequisites

- Docker and Docker Compose installed
- Ollama installed and running locally
- Mistral model downloaded in Ollama (`ollama pull mistral:instruct`)

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sayon999-d/Local-Knowledge-Chatbot.git
   cd Local-Knowledge-Chatbot/corpus/Docker
   ```

2. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

3. **Pull the required Ollama model**:
   ```bash
   ollama pull mistral:instruct
   ```

4. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

5. **Access the application**:
   Open your browser and navigate to `http://localhost:8080`

### Manual Setup (Without Docker)

1. **Install Python dependencies**:
   ```bash
   cd corpus/Docker
   pip install -r requirements.txt
   ```

2. **Set up environment variables** (optional):
   Create a `.env` file:
   ```
   OLLAMA_BASE_URL=http://localhost:11434
   ```

3. **Run the application**:
   ```bash
   streamlit run main.py --server.port=8501
   ```

4. **Access the application**:
   Open your browser and navigate to `http://localhost:8501`

## Usage

### Adding Knowledge

#### Method 1: Web URL
1. Click on "Add Knowledge" in the sidebar
2. Select "Web URL"
3. Enter a website URL
4. Click "Scrape and Learn"
5. The system will scrape the content and add it to the knowledge base

#### Method 2: File Upload
1. Click on "Add Knowledge" in the sidebar
2. Select "File Upload"
3. Upload a text file (`.txt`)
4. Click "Read and Learn"
5. The content will be processed and added to the knowledge base

### Querying the Knowledge Base

1. Type your question in the chat input at the bottom
2. The system will:
   - Search the vector database for relevant context
   - Generate an answer using the retrieved context
   - Display sources used for the answer

## Project Structure

```
Local-Knowledge-Chatbot/
├── corpus/
│   └── Docker/
│       ├── main.py                 # Main application file
│       ├── requirements.txt         # Python dependencies
│       ├── Dockerfile              # Docker image configuration
│       ├── docker-compose.yml      # Docker Compose configuration
│       ├── chroma_db_data/         # Vector database storage
│       ├── local_embeddings/       # Local embedding models
│       ├── saved_articles/         # Saved scraped articles
│       └── corpus/                 # Additional corpus data
└── README.md                       # This file
```

## Configuration

### Environment Variables

- `OLLAMA_BASE_URL`: URL for Ollama service (default: `http://host.docker.internal:11434`)
- `OLLAMA_MODEL`: Model name to use (default: `mistral:instruct`)
- `EMBEDDING_MODEL`: Embedding model name (default: `mxbai-embed-large`)

### Default Knowledge Sources

The system comes pre-configured with a list of AMD Radeon and DirectML documentation URLs. You can modify the `URL_LIST` in `main.py` to customize the initial knowledge base.

## Technical Details

### Embedding Models

The system supports multiple embedding models:
- `mxbai-embed-large`: Default large embedding model
- `all-MiniLM-L6-v2`: Smaller, faster alternative

Models are stored locally in `local_embeddings/` directory.

### Vector Database

ChromaDB is used for vector storage and retrieval. The database is persisted in `chroma_db_data/` directory.

### Text Splitting

Documents are split using `RecursiveCharacterTextSplitter` with:
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

## Troubleshooting

### Ollama Connection Issues

If you encounter connection issues with Ollama:
1. Ensure Ollama is running: `ollama serve`
2. Check the `OLLAMA_BASE_URL` environment variable
3. For Docker, ensure `host.docker.internal` is accessible

### Model Download Issues

If embedding models fail to download:
1. Check internet connection (required for initial download)
2. Ensure sufficient disk space
3. Models will be cached locally after first download

### Port Conflicts

If port 8080 (or 8501) is already in use:
- Modify the port in `docker-compose.yml` or the Streamlit command
- Update the port mapping: `"NEW_PORT:8080"`

## Development

### Adding New Features

The main application logic is in `main.py`. Key components:
- `RAGSystem`: Core RAG engine class
- `get_engine()`: Cached Streamlit resource
- UI components: Streamlit interface

### Testing

To test the system:
1. Start the application
2. Add some knowledge (URL or file)
3. Query the knowledge base
4. Verify responses and source citations

## License

This project is open source and available for personal and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on the GitHub repository.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- UI powered by [Streamlit](https://streamlit.io/)
- Vector storage by [ChromaDB](https://www.trychroma.com/)
- LLM inference by [Ollama](https://ollama.ai/)

