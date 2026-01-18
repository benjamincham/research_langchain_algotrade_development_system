import sys
from loguru import logger
from src.config.settings import settings

def setup_logging():
    """Configure loguru logging."""
    # Remove default handler
    logger.remove()
    
    # Add stdout handler
    logger.add(
        sys.stdout,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        colorize=True
    )
    
    # Add file handler for persistent logs
    log_file = settings.PROJECT_ROOT / "logs" / "system.log"
    logger.add(
        log_file,
        format=settings.LOG_FORMAT,
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="1 week"
    )
    
    return logger

# Initialize logger
logger = setup_logging()
