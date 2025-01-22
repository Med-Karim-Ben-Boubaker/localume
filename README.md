# Localume

<a href="https://www.python.org/downloads/" target="_blank"><img src="https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square" alt="Python Version: 3.8+"></a>
<a href="https://shields.io/badge/windows-11-informational?style=flat-square" target="_blank"><img src="https://shields.io/badge/windows-11-informational?style=flat-square" alt="OS: Windows 11"></a>
<a href="https://shields.io/badge/version-v1.0.0-informational?style=flat-square" target="_blank">
    <img src="https://shields.io/badge/version-v1.0.0-informational?style=flat-square" alt="Version: v1.0.0"></a>

Localume A modern desktop application for semantic file search and real-time monitoring using vector embeddings and LLM-powered search optimization.

![Project Bunner](images/bunner.png)

## Description

Localume is a powerful desktop application that enables semantic search across your documents using advanced vector embeddings and retrieval technology. The application monitors specified directories in real-time, automatically indexing new and modified files to maintain an up-to-date searchable database.
### Key Features

- Semantic Search: Find documents based on meaning, not just keywords

- Real-time Monitoring: Automatic indexing of new and modified files

- Multiple File Support: Handles PDF, TXT, and Markdown files

- Modern UI: Clean, intuitive interface with light/dark theme support

- System Tray Integration: Runs quietly in the background

- LLM-Powered: Query optimization using Google's Gemini API (Optional)

## Installation

### Prerequisites

- Python 3.8+

- Windows 11 (64-bit) (tested on windows for the moment)

- Google Gemini API key(Optional)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart-file-search.git
cd smart-file-search
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:
```bash
cd ui
python3 gui_app.py
```

2. Add folders to monitor:
	- Click "Add Folder" button
	- Select directories containing documents to index
	- The application will begin scanning and indexing files

3. Search for documents:
	- Enter your search query in natural language
	- Results will be displayed with relevant metadata
	- Double-click results to open files

### Code Examples

The application uses a modular architecture. Here's an example of performing a search:
```python
from core.search.search_engine import SearchEngine
from core.embeddings.embedding_generator import EmbeddingModel
from core.embeddings.vector_store import VectorStore

# Initialize components
vector_store = VectorStore(dimension=384)
embedding_model = EmbeddingModel()
search_engine = SearchEngine(vector_store, embedding_model)

# Perform search
results = search_engine.search("documents about project requirements", top_k=10)
```

## Architecture

The project follows a clean, modular architecture:
```txt
smart-file-search/
├── core/
│   ├── embeddings/     # Vector embedding generation and storage
│   ├── scanner/        # File monitoring and content extraction
│   ├── search/         # Search engine implementation
│   ├── llm/            # LLM service integration
│   └── utils/          # Shared utilities
│
├── data/
│   ├── faiss.index     # stores document vector embeddings
│   └── id_map.pkl      # stores a mapping between document IDs and their metadata
│
├── ui/                 # Desktop GUI implementation
```
## Contributing

Contributions are welcome! 
### Development Guidelines

- Follow PEP 8 style guide
- Add type hints to all functions
- Include docstrings for classes and methods
- Write unit tests for new features
- Update documentation as needed

## License

This project is licensed under the GNU General Public License (GPL-3.0) License - see the LICENSE file for details.

## Acknowledgments

- Azure Theme for the modern UI
- sentence-transformers for text embeddings
- Google's Gemini API for query optimization
- Facebook vector database FAISS
## Contact

 - [Mohamed Karim Ben Boubaker | LinkedIn](https://www.linkedin.com/in/mohamed-karim-ben-boubaker/) 

## FAQ

Q: How does the semantic search work?

A: The application converts documents into vector embeddings using sentence transformers, enabling similarity-based search that understands context and meaning rather than just matching keywords.

Q: What file types are supported?

A: Currently, the application supports PDF, TXT, and Markdown files. Support for additional file types is planned for future releases.

---

Note: This project is under active development. Features and documentation may be updated frequently.

