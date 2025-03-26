# src/config.py
"""
Configuration settings for QmiracTM AI Knowledge Base

This module contains all configuration settings for the knowledge base,
including file paths, model settings, and performance parameters.
"""
import os
import platform
from pathlib import Path

# Project base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
PDF_DIR = os.path.join(DATA_DIR, 'pdf')
CSV_DIR = os.path.join(DATA_DIR, 'csv')
JSON_DIR = os.path.join(DATA_DIR, 'json')
TXT_DIR = os.path.join(DATA_DIR, 'txt')
DB_PATH = os.path.join(BASE_DIR, 'database', 'kb.db')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')

# Ensure all directories exist
for directory in [PDF_DIR, CSV_DIR, JSON_DIR, TXT_DIR, os.path.dirname(DB_PATH), BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder:reasoning"  # Default model for general tasks
REASONING_MODEL = "deepseek-coder:reasoning"  # Model for complex reasoning tasks
STRATEGY_MODEL = "deepseek-coder:reasoning"  # Model for strategy generation
FALLBACK_MODEL = "llama3"  # Fallback model if primary not available

# Embedding model settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformers model
EMBEDDING_DIMENSION = 384  # Dimension of the embedding vectors

# RAG settings
CHUNK_SIZE = 500  # Default chunk size in words
CHUNK_OVERLAP = 50  # Default overlap between chunks
TOP_K_RESULTS = 5  # Default number of results to retrieve
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for retrieval

# Document type-specific chunk sizes
CHUNK_SIZES = {
    "strategy": 750,       # Strategy documents need more context
    "execution": 650,      # Execution documents need substantial context
    "market_assessment": 600,
    "financial": 400,      # Financial data prefers precision
    "data_table": 300,     # Tables work best with smaller chunks
    "general": 500         # Default for other document types
}

# Performance settings
BATCH_SIZE = 8  # Batch size for embedding generation
MAX_TOKENS = 4096  # Maximum tokens for generated responses
TEMPERATURE_SETTINGS = {
    "strategy": 0.4,   # Lower temperature for more focused strategy generation
    "query": 0.7,      # Higher temperature for more creative responses to queries
    "summary": 0.5     # Balanced for summaries
}

# Business-specific settings
RISK_TOLERANCE_LEVELS = ["High", "Medium", "Low"]
BUSINESS_DOMAINS = [
    "strategy", "execution", "market_assessment", "financial", 
    "operations", "sales", "marketing", "hr", "risk"
]

# System settings
LOGGING_ENABLED = True
DEBUG_MODE = os.environ.get("QMIRAC_DEBUG", "").lower() in ["true", "1", "yes"]
SYSTEM_INFO = {
    "os": platform.system(),
    "python_version": platform.python_version(),
    "architecture": platform.architecture()[0]
}

# Try to use multiple CPU cores if available
try:
    import multiprocessing
    CPU_COUNT = multiprocessing.cpu_count()
except (ImportError, NotImplementedError):
    CPU_COUNT = 2  # Default to 2 if we can't determine