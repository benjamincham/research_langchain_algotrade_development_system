import json
from typing import Any, Dict
from src.core.logging import logger

def save_json(data: Dict[str, Any], file_path: str):
    """Save dictionary to JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.debug(f"Saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        raise

def format_currency(amount: float, currency: str = "USD") -> str:
    """Format float as currency string."""
    return f"{currency} {amount:,.2f}"
