import tkinter as tk
from tkinter import ttk
import os
import sys
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import subprocess
import platform
from PIL import Image, ImageTk
import pystray
from pystray import MenuItem as item
import json
import hashlib

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import backend components
from core.scanner.monitor import FileSystemMonitor
from core.scanner.file_scanner import FileScanner, ScanResult
from core.search.search_engine import SearchEngine
from core.embeddings.vector_store import VectorStore, SearchResult
from core.embeddings.embedding_generator import EmbeddingModel
from core.llm.service import GeminiService, GeminiConfig

class GUIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("File Search")
        self.root.geometry("1200x800")
        
        # Flag to track if we're minimizing to tray
        self.minimizing_to_tray = False
        self.is_running = True  # Add flag to track application state
        
        # Create a queue for thread-safe GUI updates
        self.status_queue = queue.Queue()
        
        # Initialize monitored directories
        self.monitored_dirs: List[str] = [
            r"C:\Users\karim\Downloads\archive2"
        ]
        
        # Load and set the Azure theme first
        self.root.tk.call("source", "azure.tcl")  
        self.root.tk.call("set_theme", "light") 
        
        # Create widgets first so we can show progress
        self.create_widgets()
        
        # Create system tray icon
        self.setup_system_tray()
        
        # Start checking the status queue
        self.check_status_queue()
        
        # Add path for the vector store
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize backend components with progress updates
        self.initialize_backend()
        
        # File monitor will be started after backend initialization

    def setup_system_tray(self):
        """Setup the system tray icon and menu"""
        # Create a simple icon (you should replace this with your own icon)
        icon_image = Image.new('RGB', (64, 64), color='blue')
        
        # Create the system tray menu
        menu = (
            item('Show/Hide', self.handle_tray_show_hide),
            item('Exit', self.handle_tray_exit)
        )
        
        # Create the system tray icon
        self.tray_icon = pystray.Icon(
            "file_search",
            icon_image,
            "File Search",
            menu
        )
        
        # Start the system tray icon in a separate thread
        self.tray_thread = threading.Thread(target=self.tray_icon.run, daemon=True)
        self.tray_thread.start()

    def handle_tray_show_hide(self, icon):
        """Handle show/hide from system tray"""
        if not self.is_running:
            return
        self.root.after(0, self._toggle_window_state)

    def handle_tray_exit(self, icon):
        """Handle exit from system tray"""
        if not self.is_running:
            return
        self.root.after(0, self.quit_application)

    def _toggle_window_state(self):
        """Internal method to toggle window state"""
        if self.root.state() == 'withdrawn':
            self.show_window()
        else:
            self.hide_window()

    def show_window(self):
        """Show the main window"""
        if not self.is_running:
            return
        self.root.deiconify()
        self.root.state('normal')
        self.root.lift()
        self.root.focus_force()

    def hide_window(self):
        """Hide the main window"""
        if not self.is_running:
            return
        self.minimizing_to_tray = True
        try:
            self.root.withdraw()
        finally:
            self.minimizing_to_tray = False

    def check_status_queue(self):
        """Check for status updates in the queue and update the GUI"""
        try:
            while True:
                message = self.status_queue.get_nowait()
                self.status_var.set(message)
                self.root.update_idletasks()
        except queue.Empty:
            pass
        finally:
            # Schedule the next check
            self.root.after(100, self.check_status_queue)

    def update_status(self, message: str):
        """Thread-safe status update"""
        self.status_queue.put(message)

    def get_file_hash(self, file_path: str) -> str:
        """Calculate a hash based on file path, size and modification time"""
        file_stat = os.stat(file_path)
        hash_string = f"{file_path}{file_stat.st_size}{file_stat.st_mtime}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def load_processed_files(self) -> dict:
        """Load the processed files tracking data"""
        if self.processed_files_path.exists():
            try:
                with open(self.processed_files_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.update_status(f"Error loading processed files data: {e}")
                return {}
        return {}

    def save_processed_files(self, processed_files: dict):
        """Save the processed files tracking data"""
        try:
            with open(self.processed_files_path, 'w') as f:
                json.dump(processed_files, f, indent=2)
        except Exception as e:
            self.update_status(f"Error saving processed files data: {e}")

    def check_files_need_processing(self, directories: List[str]) -> List[str]:
        """Check which files need to be processed"""
        files_to_process = []
        processed_files = self.load_processed_files()
        
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    current_hash = self.get_file_hash(file_path)
                    
                    # Check if file needs processing
                    if file_path not in processed_files or processed_files[file_path] != current_hash:
                        files_to_process.append(file_path)
        
        return files_to_process

    def initialize_backend(self):
        """Initialize all backend components with progress feedback"""
        def init_backend():
            try:
                self.update_status("Initializing backend components...")
                
                # Create data directory if it doesn't exist
                self.data_dir.mkdir(exist_ok=True)
                
                # Initialize vector store
                self.vector_store = VectorStore(
                    dimension=384,
                    index_path=self.data_dir / "faiss.index",
                    id_map_path=self.data_dir / "id_map.pkl"
                )
                
                # Initialize embedding model
                self.embedding_model = EmbeddingModel()
                
                # Initialize scanner with progress callback
                self.scanner = FileScanner(
                    self.vector_store, 
                    self.embedding_model,
                    progress_callback=self.update_status
                )
                
                # Initialize Gemini service
                self.gemini_service = GeminiService(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    config=GeminiConfig()
                )
                
                # Initialize search engine
                self.search_engine = SearchEngine(
                    vector_store=self.vector_store,
                    embedding_model=self.embedding_model,
                    gemini_service=self.gemini_service
                )

                print("Performing initial parallel scan...")
                initial_scan = self.scanner.scan_directories_parallel(self.monitored_dirs)

                # Start file monitor - let it handle all file operations
                self.root.after(0, self.start_file_monitor)
                self.update_status("Ready for search.")
                
            except Exception as e:
                self.update_status(f"Error during initialization: {str(e)}")
                raise

        # Run initialization in a separate thread
        init_thread = threading.Thread(target=init_backend, name="InitThread")
        init_thread.start()

    def start_file_monitor(self):
        """Start the file system monitor in a separate thread"""
        try:
            # Create a callback that updates status
            def monitor_callback(file_path: str, event_type: str):
                self.update_status(f"File {event_type}: {file_path}")
            
            self.monitor = FileSystemMonitor(
                self.monitored_dirs, 
                self.scanner,
                self.vector_store,
                callback=monitor_callback
            )

            self.monitor_thread = threading.Thread(
                target=self.monitor.start,
                daemon=True,
                name="MonitorThread"
            )

            self.monitor_thread.start()
            self.update_status("File monitor started successfully")
        except Exception as e:
            self.update_status(f"Error starting file monitor: {str(e)}")

    def format_search_result(self, result: SearchResult) -> tuple:
        """Format a search result for the treeview"""
        metadata = result.metadata
        return (
            metadata['filename'],
            metadata.get('file_type', 'Unknown'),
            metadata['last_modified'],
            f"{metadata.get('size_bytes', 'N/A')} bytes"
        )

    def perform_search(self):
        """Execute search and update results"""
        query = self.search_var.get()
        if not query.strip():
            return
            
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        try:
            # Get total count of embeddings in vector store
            total_embeddings = self.vector_store.get_total_count()
            self.update_status(f"Searching through {total_embeddings} indexed files...")
            
            # Perform vector search
            self.current_results = self.search_engine.search(query, top_k=10)
            
            if not self.current_results:
                # Show "No results found" in the tree
                self.tree.insert("", tk.END, values=("No results found", "", "", ""))
                self.update_status("No results found")
            else:
                # Add results to tree and verify files exist
                valid_results = []
                for result in self.current_results:
                    file_path = result.metadata.get("file_path")
                    if file_path and os.path.exists(file_path):
                        valid_results.append(result)
                        formatted_result = self.format_search_result(result)
                        self.tree.insert("", tk.END, values=formatted_result)
                
                if not valid_results:
                    self.tree.insert("", tk.END, values=("No results found", "", "", ""))
                    self.update_status("No valid results found (files may have been moved or deleted)")
                else:
                    self.current_results = valid_results
                    self.update_status(f"Found {len(valid_results)} results")
                    
        except Exception as e:
            error_msg = f"Error during search: {str(e)}"
            self.update_status(error_msg)
            self.tree.insert("", tk.END, values=(error_msg, "", "", ""))

    def on_result_double_click(self, event):
        """Handle double-click on search result"""
        # Get the selected item
        selection = self.tree.selection()
        if not selection:
            return
            
        # Get the index of the selected item
        item_id = selection[0]
        item_index = self.tree.index(item_id)
        
        # Check if we have results and the index is valid
        if not hasattr(self, "current_results") or not self.current_results or item_index >= len(self.current_results):
            return
            
        # Get the file path from the selected result
        result = self.current_results[item_index]
        file_path = result.metadata.get("file_path")
        
        if not file_path:
            self.update_status("Error: File path not found in result metadata")
            return
            
        try:
            # Open the file with the default system application
            if platform.system() == "Windows":
                os.startfile(file_path)
            else:
                subprocess.run(["xdg-open", file_path], check=True)
            self.update_status(f"Opened file: {file_path}")
        except Exception as e:
            self.update_status(f"Error opening file: {str(e)}")

    def create_widgets(self):
        # Main container with Card style
        main_frame = ttk.Frame(self.root, style="Card.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(
            header_frame,
            text="File Search",
            font=("Segoe UI", 24, "bold")
        )
        title_label.pack(side=tk.LEFT)

        # Status bar
        self.status_var = tk.StringVar(value="Initializing...")
        status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10),
            foreground="#666666"
        )
        status_label.pack(fill=tk.X, pady=(0, 10))

        # Search frame
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Search label
        search_label = ttk.Label(
            search_frame,
            text="What are you searching for?",
            font=("Segoe UI", 11)
        )
        search_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Search input container
        search_input_frame = ttk.Frame(search_frame)
        search_input_frame.pack(fill=tk.X)
        
        # Search entry
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(
            search_input_frame,
            textvariable=self.search_var,
            font=("Segoe UI", 11)
        )
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        # Search button using Accent style
        search_btn = ttk.Button(
            search_input_frame,
            text="Search",
            style="Accent.TButton",
            command=self.perform_search
        )
        search_btn.pack(side=tk.RIGHT)
        
        # Bind Enter key to search
        search_entry.bind("<Return>", lambda e: self.perform_search())

        # Results area
        self.create_results_area(main_frame)

        # Add theme toggle button
        theme_button = ttk.Button(
            header_frame,
            text="Toggle Theme",
            style="Toggle.TButton",
            command=self.toggle_theme
        )
        theme_button.pack(side=tk.RIGHT, padx=10)

    def create_results_area(self, parent):
        # Results treeview
        self.tree = ttk.Treeview(
            parent,
            columns=("name", "type", "modified", "size"),
            show="headings"
        )
        
        # Configure columns
        columns = {
            "name": ("Name", 500),
            "type": ("Type", 200),
            "modified": ("Last Modified", 200),
            "size": ("Size", 100)
        }
        
        for col, (heading, width) in columns.items():
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="w")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Bind double-click event
        self.tree.bind("<Double-1>", self.on_result_double_click)
        
        # Pack the treeview and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def toggle_theme(self):
        # Get current theme
        current_theme = self.root.tk.call("ttk::style", "theme", "use")
        
        # Toggle between light and dark themes
        if current_theme == "azure-dark":
            self.root.tk.call("set_theme", "light")
        else:
            self.root.tk.call("set_theme", "dark")

    def quit_application(self):
        """Completely quit the application"""
        self.is_running = False  # Set flag to prevent further window operations
        
        try:
            # Stop the file monitor
            if hasattr(self, 'monitor'):
                self.monitor.stop()
            if hasattr(self, 'monitor_thread'):
                self.monitor_thread.join(timeout=1.0)
            
            # Stop the tray icon
            if hasattr(self, 'tray_icon'):
                self.tray_icon.stop()
            
            # Destroy the root window and quit
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            # Ensure the application exits
            os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        app = GUIApp(root)
        root.protocol("WM_DELETE_WINDOW", app.hide_window)  # Hide instead of close
        root.mainloop()
    except Exception as e:
        print(f"Error in main loop: {e}")
        sys.exit(1)

