import os
import logging
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.WHITE,
        'INFO': Fore.CYAN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, '')
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

def setup_logger(name):
    """Configures and returns a logger with colored console output and optional file logging.
    
    Args:
        name (str): Logger name (usually __name__)
        
    Returns:
        logging.Logger: Configured logger instance
        
    Environment Variables:
        LOG_LEVEL: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FILE: Optional path to log file (if not set, file logging is disabled)
        
    Notes:
        - Console output uses colored formatting
        - File logging (if enabled) uses plain text format
        - Log level defaults to INFO if not specified
    """

    load_dotenv()
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE')
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # Add file handler only if LOG_FILE is set
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    # Show log by default
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    
    return logger