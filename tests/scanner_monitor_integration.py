import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from core.scanner.monitor import FileSystemMonitor
from core.scanner.file_scanner import FileScanner
import time
from pathlib import Path

def main():
    # Directories to monitor/scan
    directories = [
        r"C:\Users\karim\Desktop",
        r"C:\Users\karim\Downloads"
    ]
    
    # Create directories if they don't exist
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Initial scan using parallel scanning
    scanner = FileScanner()
    print("Performing initial parallel scan...")
    initial_metadata = scanner.scan_directories_parallel(directories)
    
    # Write initial scan results
    scanner.write_scan_results(initial_metadata, directories, "initial_scan.log")
    
    # Start monitoring for changes
    print("Starting file system monitor...")
    monitor = FileSystemMonitor(directories)
    
    try:
        monitor.start()  # This will run until interrupted
    except KeyboardInterrupt:
        print("\nStopping monitor...")
    finally:
        # Perform final parallel scan
        print("Performing final parallel scan...")
        final_metadata = scanner.scan_directories_parallel(directories)
        scanner.write_scan_results(final_metadata, directories, "final_scan.log")

if __name__ == "__main__":
    main()