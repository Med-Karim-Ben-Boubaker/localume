import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import logging
from typing import List, Callable
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime
import threading

from .file_scanner import FileScanner, ScanResult, ScannedFile
from ..utils.logger import Logger
from ..embeddings.vector_store import VectorStore
class FileSystemMonitor:
    """
    Monitors file system changes and integrates with FileScanner to handle events.
    """
    def __init__(self, paths: List[str], file_scanner: FileScanner, vector_store: VectorStore, callback: Callable[[str, str], None]):
        self.paths = [Path(p).resolve() for p in paths]
        self.file_scanner = file_scanner
        self.vector_store = vector_store
        self.callback = callback  # Store the callback
        self.observer = Observer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = Logger("FileSystemMonitor").logger
        self._is_running = False
        self._observer_thread = None

    def start(self):
        """Start monitoring the specified directories."""
        self._is_running = True
        event_handler = self.EventHandler(
            self.file_scanner, 
            self.logger, 
            self.vector_store,
            self.callback
        )
        
        for path in self.paths:
            if path.exists() and path.is_dir():
                self.observer.schedule(event_handler, str(path), recursive=True)
                self.logger.info(f"Started monitoring: {path}")
            else:
                self.logger.error(f"Path does not exist or is not a directory: {path}")
        
        self.observer.start()
        self._observer_thread = threading.Thread(target=self._run_observer, daemon=True)
        self._observer_thread.start()

    def _run_observer(self):
        """Run the observer in a separate thread."""
        try:
            while self._is_running:
                time.sleep(1)
        except Exception as e:
            self.logger.error(f"Observer thread error: {e}")
        finally:
            self.observer.stop()
            self.observer.join()

    def stop(self):
        """Stop monitoring all directories."""
        self._is_running = False
        if self._observer_thread:
            self._observer_thread.join(timeout=2.0)
        self.observer.stop()
        self.observer.join()
        self.executor.shutdown(wait=False)
        self.logger.info("File system monitor stopped")

    def update_directories(self, new_paths: List[str]):
        """Update the list of monitored directories."""
        # Stop current monitoring
        self.observer.unschedule_all()
        
        # Update paths
        self.paths = [Path(p).resolve() for p in new_paths]
        
        # Create new event handler
        event_handler = self.EventHandler(
            self.file_scanner, 
            self.logger, 
            self.vector_store,
            self.callback
        )
        
        # Schedule monitoring for new paths
        for path in self.paths:
            if path.exists() and path.is_dir():
                self.observer.schedule(event_handler, str(path), recursive=True)
                self.logger.info(f"Updated monitoring for: {path}")
            else:
                self.logger.error(f"Path does not exist or is not a directory: {path}")

    def remove_directory_from_index(self, directory_path: str) -> None:
        """
        Remove all files from a directory from the vector store index.
        
        Args:
            directory_path: Path to the directory whose files should be removed
            
        Raises:
            ValueError: If the directory path is invalid
            RuntimeError: If there's an error removing files from the index
        """
        try:
            directory = Path(directory_path).resolve()
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Invalid directory path: {directory_path}")
            
            # Get all file IDs in this directory from vector store
            removed_count = 0
            errors = []
            
            # Batch process files for better performance
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    try:
                        # Generate the unique ID used when the file was indexed
                        file_id = self.file_scanner.generate_unique_id(str(file_path))
                        self.vector_store.remove_embedding(file_id)
                        removed_count += 1
                        
                        # Notify through callback if provided
                        if self.callback:
                            self.callback(str(file_path), "removed")
                            
                    except Exception as e:
                        errors.append((str(file_path), str(e)))
                        self.logger.error(f"Error removing file {file_path} from index: {e}")
            
            # Log summary
            self.logger.info(f"Removed {removed_count} files from index in {directory_path}")
            if errors:
                error_msg = f"Encountered {len(errors)} errors while removing files"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg, errors)
            
        except Exception as e:
            self.logger.error(f"Error removing directory from index: {e}")
            raise

    class EventHandler(FileSystemEventHandler):
        """Handles file system events and delegates processing to FileScanner."""
        
        def __init__(self, file_scanner: FileScanner, logger: logging.Logger, 
                     vector_store: VectorStore, callback: Callable[[str, str], None]):
            super().__init__()
            self.file_scanner = file_scanner
            self.logger = logger
            self.vector_store = vector_store
            self.callback = callback  # Store the callback
            self.executor = ThreadPoolExecutor(max_workers=4)
            self._last_modified = {}
            self._cooldown = 2  # seconds
            self.supported_extensions = file_scanner.SUPPORTED_EXTENSIONS
            self._recently_created = set()

        def is_supported_file(self, path: str) -> bool:
            """Check if the file is supported by the scanner."""
            file_extension = os.path.splitext(path)[1].lower()
            return file_extension in self.supported_extensions

        def _should_ignore(self, event) -> bool:
            """Return True if the event should be ignored."""
            path = event.src_path
            
            # Ignore directories
            if event.is_directory:
                return True
                
            # Ignore system and temporary files
            ignore_patterns = [
                'desktop.ini',
                'Thumbs.db',
                '.tmp',
                '.temp',
                '~$',  # Temporary Office files
                '.crdownload',  # Chrome downloads
                '.part'  # Partial downloads
            ]
            
            filename = os.path.basename(path)
            return any(pattern in filename for pattern in ignore_patterns)

        def process_event(self, path: str, event_type: str):
            """Process the file system event by updating metadata."""
            try:
                if not self.is_supported_file(path):
                    return

                file_path = Path(path)
                if not file_path.exists():
                    return

                time.sleep(0.5)

                scanned_file = self.file_scanner.scan_file(str(file_path))

                # If scan was successful (has embedding and metadata)
                if scanned_file.embedding is not None and scanned_file.metadata:
                    # Create scan result for logging
                    scan_result = ScanResult(
                        scanned_files=[scanned_file],
                        scan_time=datetime.now(),
                        scanned_paths=[str(file_path.parent)],
                        errors=[]
                    )

                    self.file_scanner.write_scan_results(scan_result)
                    
                    # Notify GUI through callback
                    if self.callback:
                        self.callback(path, event_type)

                    self.logger.info(f"Successfully processed {event_type} event for: {path}")
                else:
                    self.logger.warning(f"No content extracted from file: {path}")

            except Exception as e:
                self.logger.error(f"Error processing {event_type} event for '{path}': {str(e)}")

        def on_created(self, event):
            """Handle file creation events."""
            if not self._should_ignore(event):
                self.logger.info(f"Created: {event.src_path}")
                self._recently_created.add(event.src_path)
                self.executor.submit(self.process_event, event.src_path, "created")
                
                def remove_from_recent():
                    time.sleep(2)  # Wait 2 seconds
                    self._recently_created.discard(event.src_path)
                
                threading.Thread(target=remove_from_recent, daemon=True).start()

        def on_modified(self, event):
            """Handle file modification events."""
            if self._should_ignore(event):
                return
            
            if event.src_path in self._recently_created:
                return
            
            current_time = time.time()
            last_modified = self._last_modified.get(event.src_path, 0)

            if current_time - last_modified > self._cooldown:
                self.logger.info(f"Modified: {event.src_path}")
                self.executor.submit(self.process_event, event.src_path, "modified")
                self._last_modified[event.src_path] = current_time

        def on_deleted(self, event):
            """Handle file deletion events."""
            if not self._should_ignore(event):
                self.logger.info(f"Deleted: {event.src_path}")
                
                # Remove embedding from vector store
                unique_id = self.file_scanner.generate_unique_id(event.src_path)
                self.vector_store.remove_embedding(unique_id)

        def on_moved(self, event):
            """Handle file move events."""
            if not self._should_ignore(event):
                self.logger.info(f"Moved: from {event.src_path} to {event.dest_path}")
                # Could implement updating paths in vector store here if needed