# src/config.py
"""
Configuration settings for QmiracTM AI Knowledge Base

This module contains all configuration settings for the knowledge base,
including file paths, model settings, and performance parameters.
"""
import os
from pathlib import Path

# Project base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
PDF_DIR = os.path.join(DATA_DIR, 'pdf')
CSV_DIR = os.path.join(DATA_DIR, 'csv')
DB_PATH = os.path.join(BASE_DIR, 'database', 'kb.db')
BACKUP_DIR = os.path.join(BASE_DIR, 'backups')

# Ensure all directories exist
for directory in [PDF_DIR, CSV_DIR, os.path.dirname(DB_PATH), BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)

# LLM settings
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "deepseek-coder:reasoning"  # Default model for general tasks
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

# Document types
DOCUMENT_TYPES = {
    "strategy": "Strategic assessments and plans",
    "execution": "Execution plans and operational data",
    "market_assessment": "Market analysis and target segments",
    "competitive": "Competitive landscape analysis",
    "financial": "Financial data and performance metrics",
    "swot": "SWOT analysis",
    "risk": "Risk assessments and mitigation",
    "data_table": "Metric data tables"
}