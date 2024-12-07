import logging
from pathlib import Path

class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
            
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False  # Prevent propagation to parent loggers

        # Create handlers
        c_handler = logging.StreamHandler()
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        f_handler = logging.FileHandler(f'logs/{name}.log')
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