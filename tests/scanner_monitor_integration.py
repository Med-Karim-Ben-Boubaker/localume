import os
import sys
import threading
from pathlib import Path
from typing import List
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from core.scanner.monitor import FileSystemMonitor
from core.scanner.file_scanner import FileScanner, ScanResult
from core.search.search_engine import SearchEngine
from core.embeddings.vector_store import VectorStore, SearchResult
from core.embeddings.embedding_generator import EmbeddingModel
from core.llm.service import GeminiService, GeminiConfig

def format_search_result(result: SearchResult, idx: int) -> str:
    """
    Format a search result for display.
    
    Args:
        result (SearchResult): The search result to format
        idx (int): Result index number
        
    Returns:
        str: Formatted search result string
    """
    return (
        f"{idx}. File: {result.metadata['filename']}\n"
        f"   Path: {result.metadata['file_path']}\n" 
        f"   Distance: {result.distance:.4f}\n"
        f"   Last Modified: {result.metadata['last_modified']}\n"
    )

def handle_user_queries(search_engine: SearchEngine):
    """
    Handle user search queries in a separate thread.
    
    Args:
        search_engine: The search engine instance to use for queries
    """
    print("\nSearch Interface Ready - Enter your queries (type 'exit' to quit):")
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            if query.lower() == 'exit':
                break
                
            # Perform vector search
            vector_results = search_engine.search(query, top_k=10)

            print("\nSearch Results:")
            if not vector_results:
                print("No matching documents found.")
            else:
                for idx, result in enumerate(vector_results, 1):
                    print(f"\n{format_search_result(result, idx)}")
                    
        except Exception as e:
            print(f"Error processing query: {str(e)}")

def print_scan_summary(scan_result: ScanResult) -> None:
    """
    Print a summary of the scan results.
    
    Args:
        scan_result (ScanResult): The scan results to summarize
    """
    print(f"\nScan completed at: {scan_result.scan_time.isoformat()}")
    print(f"Scanned directories: {', '.join(scan_result.scanned_paths)}")
    print(f"Files processed: {len(scan_result.scanned_files)}")
    
    if scan_result.errors:
        print("\nErrors encountered:")
        for error in scan_result.errors:
            print(f"- {error}")

def main():
    # Directories to monitor/scan
    directories: List[str] = [
        r"C:\Users\karim\Downloads\archive2"  # Adjust this path as needed
    ]
    
    # Create directories if they don't exist
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Initialize components with proper paths
    vector_store = VectorStore(
        dimension=384,  # Dimension for all-MiniLM-L6-v2 model
        index_path=data_dir / "faiss.index",
        id_map_path=data_dir / "id_map.pkl"
    )
    
    embedding_model = EmbeddingModel()
    scanner = FileScanner(vector_store, embedding_model)

    gemini_service = GeminiService(
        api_key=os.getenv("GEMINI_API_KEY"),
        config=GeminiConfig()
    )

    # Initialize search engine with Gemini service
    search_engine = SearchEngine(
        vector_store=vector_store,
        embedding_model=embedding_model,
        gemini_service=gemini_service
    )
    
    # Perform initial parallel scan
    print("Performing initial parallel scan...")
    initial_scan = scanner.scan_directories_parallel(directories)
    print_scan_summary(initial_scan)
    
    # Write initial scan results
    scanner.write_scan_results(initial_scan, "initial_scan.log")
    
    # Start monitoring for changes
    print("\nStarting file system monitor...")
    monitor = FileSystemMonitor(
        directories,
        scanner,
        vector_store,
        None)
    
    # Create and start the query handling thread
    query_thread = threading.Thread(
        target=handle_user_queries,
        args=(search_engine,),
        daemon=True
    )
    query_thread.start()
    
    try:
        monitor.start()  # This will run until interrupted
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        # Wait for query thread to finish     
        query_thread.join(timeout=1.0)
        print("Shutdown complete.")

if __name__ == "__main__":
    main()