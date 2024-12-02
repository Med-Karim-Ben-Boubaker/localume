import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Iterator
from datetime import datetime
import mimetypes
import logging

class FileScanner:
    """
    A class to scan directories, extract metadata, and write scan results.
    """
    def __init__(self):	
        self.setup_logger()
    
    def setup_logger(self):
        """
        Set up the logger for the FileScanner.
        """
        self.logger = logging.getLogger("FileScanner")
        self.logger.setLevel(logging.DEBUG)

        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('logs/file_scanner.log')
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

    def get_file_metadata(self, entry: os.DirEntry) -> List[Dict[str, Any]]:
        """
        Extract metadata from a file
        
        Args:
            entry: DirEntry object representing the file
        Returns:
            Dictionary containing file metadata
        """
        try:
            stats = entry.stat()
            mime_type, _ = mimetypes.guess_type(entry.path)
            
            created_time = datetime.fromtimestamp(stats.st_birthtime)
            
            # return metadata in the form of dictionary
            return {
                "name": entry.name,
                "path": entry.path,
                "size_bytes": stats.st_size,
                "size_mb": round(stats.st_size / (1024 * 1024), 2),
                "created_time": created_time.strftime('%Y-%m-%d %H:%M:%S'),
                "modified_time": datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                "accessed_time": datetime.fromtimestamp(stats.st_atime).strftime('%Y-%m-%d %H:%M:%S'),
                "file_type": mime_type or "unknown",
                "is_file": entry.is_file(),
                "is_dir": entry.is_dir(),
                "is_symlink": entry.is_symlink()
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting metadata from '{entry.path}': {str(e)}")
            return {
                "name": entry.name,
                "path": entry.path,
                "error": str(e)
            }
    
    def scan_directories_parallel(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Scan multiple directories in parallel and collect metadata.
        """
        all_metadata = []
        
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            # Submit scan tasks
            futures = {executor.submit(self.scan_directory, path): path for path in paths}
            
            # Iterate over completed futures
            for future in as_completed(futures):
                path = futures[future]
                try:
                    metadata = future.result()
                    all_metadata.extend(metadata)
                except Exception as e:
                    self.logger.error(f"Error scanning '{path}': {str(e)}")
        
        return all_metadata

    def scan_directory(self, root_path: str) -> List[Dict[str, Any]]:
        """
        Recursively scan directories and files starting from root_path using depth-first search.
        
        Args:
            root_path (str): The starting directory path to scan
        """
        all_metadata = []

        try:
            # Convert the root_path to absolute path and resolve any symbolic links
            root = Path(root_path).resolve()
            
            # Check if the path exists
            if not root.exists():
                self.logger.error(f"Error: Path '{root}' does not exist")
                return all_metadata
                
            # Use os.scandir() to iterate through directory entries
            with os.scandir(root) as entries:
                for entry in entries:
                    try:
                        # Get absolute path
                        abs_path = entry.path

                        # Get file metadata
                        file_metadata = self.get_file_metadata(entry)
                        all_metadata.append(file_metadata)

                        # If entry is a directory, recursively scan it
                        if entry.is_dir():
                            self.scan_directory(abs_path)
                            
                        # If entry is a file, print its details
                        elif entry.is_file():
                            # Get file size in bytes
                            size = entry.stat().st_size

                    except PermissionError:
                        self.logger.warning(f"Permission denied: Cannot access '{entry.path}'")
                    except Exception as e:
                        self.logger.error(f"Error processing '{entry.path}': {str(e)}")

        except Exception as e:
            self.logger.error(f"Error scanning '{root}': {str(e)}")

        return all_metadata

    def write_scan_results(self, metadata_iterator: Iterator[Dict[str, Any]], paths: List[str], output_file: str = "scan_result.log") -> None:
        """
        Write scanning results to a file.
        """
        try:
            # Ensure logs directory exists
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            output_path = logs_dir / output_file
            
            with output_path.open('w', encoding='utf-8') as f:
                # Write header
                f.write("File System Scan Results\n\n")
                f.write(f"{'='*50}\n")
                f.write(f"Scan started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Scanned directories:\n")
                for path in paths:
                    f.write(f"- {path}\n")
                f.write(f"{'='*50}\n\n")

                # Write metadata for each file
                for metadata in metadata_iterator:
                    for key, value in metadata.items():
                        f.write(f"{key}: {value}\n")
                    f.write("\n")
    
        except Exception as e:
            self.logger.error(f"Error writing to '{output_path}': {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize FileScanner instance
    file_scanner = FileScanner()

    # Specify your root directory here
    root_directory = r"C:\Users\karim"  # Change this to your desired path

    directories = [
        r"C:\Users\karim\Documents",
        r"C:\Users\karim\Downloads",
        r"C:\Users\karim\Music",
        r"C:\Users\karim\Pictures",
        r"C:\Users\karim\Videos",
        r"C:\Users\karim\Desktop"
    ]

    
    scanned_metadata = file_scanner.scan_directories_parallel(directories)
    file_scanner.write_scan_results(scanned_metadata, directories)


