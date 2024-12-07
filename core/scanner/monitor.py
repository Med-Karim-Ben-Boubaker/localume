import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pathlib import Path
import logging
from typing import List
from concurrent.futures import ThreadPoolExecutor
from .file_scanner import FileScanner
import os

class FileSystemMonitor:
    """
    Monitors file system changes and integrates with FileScanner to handle events.
    """
    def __init__(self, paths: List[str], file_scanner: FileScanner):
        self.paths = [Path(p).resolve() for p in paths]
        self.file_scanner = file_scanner
        self.observer = Observer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.setup_logger()

    def setup_logger(self):
        """
        Set up the logger for the FileSystemMonitor.
        """
        self.logger = logging.getLogger("FileSystemMonitor")
        self.logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/monitor.log')
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add to handlers
        c_format = logging.Formatter('%(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

    def start(self):
        """
        Start monitoring the specified directories.
        """
        event_handler = self.EventHandler(self.file_scanner, self.logger)
        for path in self.paths:
            if path.exists() and path.is_dir():
                # recursive=True to monitor subdirectories
                self.observer.schedule(event_handler, str(path), recursive=True)
                self.logger.info(f"Started monitoring: {path}")
            else:
                self.logger.error(f"Path does not exist or is not a directory: {path}")
        
        self.observer.start()
        try:
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Stopping file system monitor...")
            self.observer.stop()

        # wait for the observer to finish
        self.observer.join()

    class EventHandler(FileSystemEventHandler):
        """
        Handles file system events and delegates processing to FileScanner.
        """
        def __init__(self, file_scanner: FileScanner, logger: logging.Logger):
            super().__init__()
            self.file_scanner = file_scanner
            self.logger = logger
            self.executor = ThreadPoolExecutor(max_workers=4)

            # Add debounce protection from continuous events (e.g. windows frequent updates)
            self._last_modified = {}
            self._cooldown = 1  # seconds

        def on_created(self, event):
            self.logger.info(f"Created: {event.src_path}")
            if not event.is_directory:
                self.executor.submit(self.process_event, event.src_path, "created")

        def on_modified(self, event):
            #Skip system files as temporary modifications
            if self._should_ignore(event):
                return
            
            current_time = time.time()
            last_modified = self._last_modified.get(event.src_path, 0)

            # Only process if enough time has passed since last modification
            if current_time - last_modified > self._cooldown:
                self.logger.info(f"Modified: {event.src_path}")
                if not event.is_directory:
                    self.executor.submit(self.process_event, event.src_path, "modified")
                self._last_modified[event.src_path] = current_time
        
        def _should_ignore(self, event):
            """Return True if the event should be ignored."""
            # Ignore desktop.ini files
            if event.src_path.endswith('desktop.ini'):
                return True
                
            # Ignore thumbnail cache files
            if 'Thumbs.db' in event.src_path:
                return True
                
            # Ignore temporary files
            if event.src_path.endswith('.tmp'):
                return True
                
            # Add more conditions as needed
            return False

        def on_deleted(self, event):
            self.logger.info(f"Deleted: {event.src_path}")

        def on_moved(self, event):
            self.logger.info(f"Moved: from {event.src_path} to {event.dest_path}")
        
        def process_event(self, path: str, event_type: str):
            """
            Process the file system event by updating metadata.
            """
            try:
                file_path = Path(path)
                if event_type in ["created", "modified"]:
                    with os.scandir(file_path.parent) as entries:
                        for entry in entries:
                            if entry.path == path:
                                metadata = self.file_scanner.get_file_metadata(entry)
                                self.file_scanner.write_scan_results([metadata], [str(file_path.parent)])
                                self.logger.info(f"Processed {event_type} event for: {path}")
                                break
            except Exception as e:
                self.logger.error(f"Error processing {event_type} event for '{path}': {str(e)}")