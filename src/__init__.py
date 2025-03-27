"""
QmiracTM AI-Driven Knowledge Base
================================

A Retrieval-Augmented Generation (RAG) system for business strategy development.

This package contains the core components of the QmiracTM RAG system:
- Database utilities for document and embedding storage
- Document loaders for PDF and CSV files
- Embedding generation for text chunks
- Retrieval mechanisms for finding relevant information
- LLM integration for generating responses
- Complete RAG pipeline for processing queries and generating strategy recommendations
"""

# Import key components for easier access
from .db import Database
from .document_loader import DocumentLoader
from .embeddings import EmbeddingGenerator
from .retriever import Retriever
from .llm import LLMManager
from .rag import RAGSystem

# Add new components
from .output_generator import StrategyOutputGenerator

# Import configuration
from .config import (
    # Directories
    BASE_DIR, DATA_DIR, PDF_DIR, CSV_DIR, JSON_DIR, TXT_DIR, DB_PATH, BACKUP_DIR,
    
    # Model settings
    OLLAMA_BASE_URL, DEFAULT_MODEL, REASONING_MODEL, STRATEGY_MODEL, FALLBACK_MODEL,
    EMBEDDING_MODEL, EMBEDDING_DIMENSION,
    
    # RAG settings
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, SIMILARITY_THRESHOLD,
    CHUNK_SIZES, BATCH_SIZE, MAX_TOKENS, TEMPERATURE_SETTINGS,
    
    # Business settings
    RISK_TOLERANCE_LEVELS, BUSINESS_DOMAINS
)

# Package metadata
__version__ = '0.1.0'
__author__ = 'QmiracTM Team'

def get_version():
    """Return the version of the package."""
    return __version__

def create_knowledge_base(db_path=DB_PATH):
    """
    Create and return a complete knowledge base system.
    
    This is a convenience function to initialize all components
    of the QmiracTM Knowledge Base at once.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Tuple containing (Database, DocumentLoader, EmbeddingGenerator, 
                         Retriever, LLMManager, RAGSystem)
    """
    # Initialize components
    db = Database(db_path)
    embedding_generator = EmbeddingGenerator()
    llm_manager = LLMManager()
    document_loader = DocumentLoader(db)
    retriever = Retriever(db, embedding_generator)
    
    # Initialize RAG system
    rag_system = RAGSystem(db, retriever, llm_manager)
    
    return db, document_loader, embedding_generator, retriever, llm_manager, rag_system