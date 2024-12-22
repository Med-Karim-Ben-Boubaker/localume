"""
A GUI application for file searching and monitoring with vector embeddings.
Provides real-time file monitoring, semantic search capabilities, and system tray integration.
"""

import os
import sys
import threading
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Dict, Tuple, Any

import tkinter as tk
from tkinter import ttk, filedialog
import platform
from PIL import Image
import pystray
from pystray import MenuItem
import subprocess

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Import backend components
from core.scanner.monitor import FileSystemMonitor
from core.scanner.file_scanner import FileScanner
from core.search.search_engine import SearchEngine
from core.embeddings.vector_store import VectorStore, SearchResult
from core.embeddings.embedding_generator import EmbeddingModel
from core.llm.service import GeminiService, GeminiConfig

@dataclass
class TreeviewColumn:
    """Configuration for a treeview column."""
    name: str
    display_name: str
    width: int

class GUIApp:
    """
    Main GUI application class that handles the user interface and coordinates
    backend components for file searching and monitoring.
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """
        Initialize the GUI application with all necessary components.
        
        Args:
            root: The root Tkinter window
        """
        self.root = root
        self.root.title("File Search")
        self.root.geometry("1200x800")
        
        # Application state
        self.monitored_dirs: List[str] = []
        self.minimizing_to_tray: bool = False
        self.is_running: bool = True
        self.current_results: List[SearchResult] = []
        
        # GUI state
        self.status_queue: queue.Queue = queue.Queue()
        self.status_var: tk.StringVar = tk.StringVar(value="Initializing...")
        self.search_var: tk.StringVar = tk.StringVar()
        
        # Initialize paths
        self.data_dir = PROJECT_ROOT / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Setup UI components
        self._load_theme()
        self._create_widgets()
        self._setup_system_tray()
        
        # Start backend services
        self._initialize_backend()
        self._start_status_checker()

    def _load_theme(self) -> None:
        """Load and configure the Azure theme."""
        self.root.tk.call("source", "azure.tcl")
        self.root.tk.call("set_theme", "light")

    def _start_status_checker(self) -> None:
        """Start the periodic status queue checker."""
        self.root.after(100, self._check_status_queue)

    def _check_status_queue(self) -> None:
        """Process status updates from the queue and update the GUI."""
        try:
            while True:
                message = self.status_queue.get_nowait()
                self.status_var.set(message)
                self.root.update_idletasks()
        except queue.Empty:
            pass
        finally:
            if self.is_running:
                self.root.after(100, self._check_status_queue)

    def update_status(self, message: str) -> None:
        """
        Thread-safe method to update the status message.
        
        Args:
            message: Status message to display
        """
        self.status_queue.put(message)

    def _setup_system_tray(self) -> None:
        """Setup the system tray icon and menu"""
        # Create a simple icon (you should replace this with your own icon)
        icon_image = Image.new('RGB', (64, 64), color='blue')
        
        # Create the system tray menu
        menu = (
            MenuItem('Show/Hide', self.handle_tray_show_hide),
            MenuItem('Exit', self.handle_tray_exit)
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

    def _initialize_backend(self) -> None:
        """Initialize backend components in a separate thread."""
        def init_worker() -> None:
            try:
                self.update_status("Initializing backend components...")
                
                # Initialize vector store
                self.vector_store = VectorStore(
                    dimension=384,
                    index_path=self.data_dir / "faiss.index",
                    id_map_path=self.data_dir / "id_map.pkl"
                )
                
                # Initialize other components
                self.embedding_model = EmbeddingModel()
                self.scanner = FileScanner(
                    self.vector_store,
                    self.embedding_model,
                    progress_callback=self.update_status
                )
                
                # Initialize LLM service
                self.gemini_service = GeminiService(
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                    config=GeminiConfig()
                )
                
                # Initialize search engine
                self.search_engine = SearchEngine(
                    vector_store=self.vector_store,
                    embedding_model=self.embedding_model,
                    gemini_service=self.gemini_service
                )

                # Perform initial scan
                self.update_status("Performing initial scan...")
                self.scanner.scan_directories_parallel(self.monitored_dirs)
                
                # Start file monitor
                self.root.after(0, self._start_file_monitor)
                self.update_status("Ready for search.")
                
            except Exception as e:
                self.update_status(f"Initialization error: {str(e)}")
                raise

        threading.Thread(target=init_worker, name="InitThread").start()

    def _start_file_monitor(self) -> None:
        """Initialize and start the file system monitor."""
        try:
            def on_file_event(file_path: str, event_type: str) -> None:
                self.update_status(f"File {event_type}: {file_path}")
            
            self.monitor = FileSystemMonitor(
                self.monitored_dirs,
                self.scanner,
                self.vector_store,
                callback=on_file_event
            )

            self.monitor_thread = threading.Thread(
                target=self.monitor.start,
                daemon=True,
                name="MonitorThread"
            )
            self.monitor_thread.start()
            
            self.update_status("File monitor started successfully")
            
        except Exception as e:
            self.update_status(f"Monitor error: {str(e)}")

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

    def _create_widgets(self) -> None:
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

        # Add Folders Section
        folders_frame = ttk.LabelFrame(main_frame, text="Monitored Folders", padding=10)
        folders_frame.pack(fill=tk.X, pady=(0, 20))

        # Listbox to show monitored directories
        self.folders_listbox = tk.Listbox(
            folders_frame,
            height=3,
            selectmode=tk.SINGLE,
            font=("Segoe UI", 10)
        )
        self.folders_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        # Scrollbar for listbox
        folders_scrollbar = ttk.Scrollbar(folders_frame, orient=tk.VERTICAL)
        folders_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        # Configure scrollbar
        self.folders_listbox.config(yscrollcommand=folders_scrollbar.set)
        folders_scrollbar.config(command=self.folders_listbox.yview)

        # Buttons frame
        folders_buttons_frame = ttk.Frame(folders_frame)
        folders_buttons_frame.pack(side=tk.LEFT, padx=(10, 0))

        # Add and Remove folder buttons
        add_folder_btn = ttk.Button(
            folders_buttons_frame,
            text="Add Folder",
            command=self.add_folder,
            style="Accent.TButton"
        )
        add_folder_btn.pack(pady=(0, 5))

        remove_folder_btn = ttk.Button(
            folders_buttons_frame,
            text="Remove Folder",
            command=self.remove_folder
        )
        remove_folder_btn.pack()

        # Status bar
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

    def add_folder(self):
        """Open folder dialog and add selected folder to monitored directories"""
        folder = filedialog.askdirectory(
            title="Select Folder to Monitor",
            mustexist=True
        )
        
        if folder:
            # Convert to absolute path and normalize
            folder = os.path.abspath(folder)
            
            # Check if folder is already monitored
            if folder not in self.monitored_dirs:
                self.monitored_dirs.append(folder)
                self.folders_listbox.insert(tk.END, folder)
                
                # If backend is already initialized, update the monitor
                if hasattr(self, 'monitor'):
                    self.monitor.update_directories(self.monitored_dirs)
                    # Scan the new directory
                    threading.Thread(
                        target=lambda: self.scanner.scan_directories_parallel([folder]),
                        daemon=True
                    ).start()
                
                self.update_status(f"Added folder: {folder}")
            else:
                self.update_status("This folder is already being monitored")

    def remove_folder(self) -> None:
        """Remove selected folder from monitoring and clean up its indexed files."""
        selection = self.folders_listbox.curselection()
        if not selection:
            self.update_status("Please select a folder to remove")
            return
            
        index = selection[0]
        folder = self.monitored_dirs[index]
        
        try:
            # Remove folder from monitoring first
            self.monitored_dirs.pop(index)
            self.folders_listbox.delete(index)
            
            # Update monitor directories
            if hasattr(self, 'monitor'):
                self.monitor.update_directories(self.monitored_dirs)
            
            # Start cleanup in background thread to keep UI responsive
            def cleanup_worker():
                try:
                    self.update_status(f"Removing indexed files from {folder}...")
                    self.monitor.remove_directory_from_index(folder)
                    self.update_status(f"Successfully removed folder: {folder}")
                except ValueError as e:
                    self.update_status(f"Invalid folder path: {str(e)}")
                except RuntimeError as e:
                    self.update_status(f"Partial failure removing folder: {str(e)}")
                except Exception as e:
                    self.update_status(f"Error removing folder: {str(e)}")
            
            threading.Thread(
                target=cleanup_worker,
                name="CleanupThread",
                daemon=True
            ).start()
            
        except Exception as e:
            self.update_status(f"Error removing folder: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    
    try:
        app = GUIApp(root)
        root.protocol("WM_DELETE_WINDOW", app.hide_window)  # Hide instead of close
        root.mainloop()
    except Exception as e:
        print(f"Error in main loop: {e}")
        sys.exit(1)

