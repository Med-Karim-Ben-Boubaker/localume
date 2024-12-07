import os
import sys
import threading
from pathlib import Path
from typing import List

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from core.scanner.monitor import FileSystemMonitor
from core.scanner.file_scanner import FileScanner
from core.search.search_engine import SearchEngine
from core.embeddings.vector_store import VectorStore
from core.embeddings.embedding_generator import EmbeddingModel

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
                
            # Perform both vector and linear search
            vector_results = search_engine.search(query, top_k=10)

            print("\nVector Search Results:")
            for idx, result in enumerate(vector_results, 1):
                print(f"{idx}. {result}")
                    
        except Exception as e:
            print(f"Error processing query: {str(e)}")

def main():
    # Directories to monitor/scan
    directories: List[str] = [
        r"C:\Users\karim\Downloads"
    ]
    
    # Create directories if they don't exist
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Set the dimension based on the embedding model output
    dimension = 768  # For distilbert-base-uncased
    vector_store = VectorStore(dimension)
    embedding_model = EmbeddingModel()
    
    # Initialize scanner
    scanner = FileScanner(vector_store, embedding_model)
    
    # Initialize search engine
    search_engine = SearchEngine(vector_store, embedding_model)
    
    # Initial scan using parallel scanning
    print("Performing initial parallel scan...")
    initial_metadata = scanner.scan_directories_parallel(directories)
    # Write initial scan results
    scanner.write_scan_results(initial_metadata, directories, "initial_scan.log")
    
    # Start monitoring for changes
    print("Starting file system monitor...")
    monitor = FileSystemMonitor(directories, scanner)
    
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
        
        # Perform final parallel scan
        print("Performing final parallel scan...")
        final_metadata = scanner.scan_directories_parallel(directories)
        scanner.write_scan_results(final_metadata, directories, "final_scan.log")

if __name__ == "__main__":
    main()