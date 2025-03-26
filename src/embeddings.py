# src/embeddings.py
import os
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Any, Tuple, Optional
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL

class EmbeddingGenerator:
    def __init__(self, model_name=EMBEDDING_MODEL):
        """
        Initialize the embedding generator with the specified model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        try:
            # Load the embedding model
            self.model = SentenceTransformer(model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            print(f"Initialized embedding model: {model_name} (dimension: {self.embedding_dimension})")
        except Exception as e:
            print(f"Error loading embedding model {model_name}: {e}")
            print("Falling back to default model: all-MiniLM-L6-v2")
            self.model_name = "all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        
    def chunk_text(self, text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
        """
        Split text into overlapping chunks using a more sophisticated approach.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in words
            overlap: Overlap between chunks in words
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return []
        
        # For business documents, try to keep sections together
        # Look for common business document section markers
        section_markers = [
            r'#+\s+[\w\s]+',  # Markdown headers
            r'\d+\.\d+\s+[\w\s]+',  # Numbered sections (e.g., "2.1 Market Analysis")
            r'[A-Z][A-Za-z\s]+:',  # Titled sections with colon (e.g., "Executive Summary:")
            r'\*\*[^*]+\*\*',  # Bold text as headers (e.g., "**Strategic Direction**")
        ]
        
        # Join patterns and find all potential section boundaries
        pattern = '|'.join(section_markers)
        
        # First try to split by sections for better semantic coherence
        if re.search(pattern, text):
            # If we find section markers, use them to split initially
            section_splits = []
            last_pos = 0
            
            for match in re.finditer(pattern, text):
                # Don't split at the very beginning
                if match.start() > 0:
                    section_splits.append(text[last_pos:match.start()].strip())
                last_pos = match.start()
            
            # Add the final section
            if last_pos < len(text):
                section_splits.append(text[last_pos:].strip())
                
            paragraphs = []
            for section in section_splits:
                if not section:
                    continue
                # Further split sections into paragraphs
                section_paragraphs = re.split(r'\n\s*\n|\.\s+', section)
                paragraphs.extend([p.strip() for p in section_paragraphs if p.strip()])
        else:
            # If no section markers found, fall back to paragraph splitting
            paragraphs = re.split(r'\n\s*\n|\.\s+', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            words = paragraph.split()
            
            if not words:
                continue
                
            # If adding this paragraph exceeds chunk size, finish the current chunk
            if current_length + len(words) > chunk_size and current_length > 0:
                chunks.append(' '.join(current_chunk))
                
                # Keep some overlap words for context
                overlap_words = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_chunk = overlap_words
                current_length = len(overlap_words)
            
            # Add paragraph to current chunk
            current_chunk.extend(words)
            current_length += len(words)
            
            # If current chunk exceeds chunk size, split it
            while current_length > chunk_size:
                chunks.append(' '.join(current_chunk[:chunk_size]))
                current_chunk = current_chunk[chunk_size-overlap:] if chunk_size-overlap > 0 else []
                current_length = len(current_chunk)
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        # Ensure no empty chunks
        chunks = [chunk for chunk in chunks if chunk.strip()]
        
        return chunks
    
    def chunk_business_document(self, text: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Chunk document with document type-specific processing.
        
        Args:
            text: Document text
            doc_type: Type of document (strategy, execution, etc.)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Default chunk size and overlap
        chunk_size = CHUNK_SIZE
        overlap = CHUNK_OVERLAP
        
        # Adjust chunking parameters based on document type
        if doc_type == "strategy":
            # Strategic documents need more context, use larger chunks
            chunk_size = min(800, CHUNK_SIZE * 1.5)
        elif doc_type == "financial" or doc_type == "data_table":
            # Financial data needs precision, use smaller chunks with less overlap
            chunk_size = min(400, CHUNK_SIZE * 0.8)
            overlap = min(30, CHUNK_OVERLAP * 0.6)
        
        # Get raw text chunks
        raw_chunks = self.chunk_text(text, chunk_size=int(chunk_size), overlap=int(overlap))
        
        # Enhance chunks with metadata
        enhanced_chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            enhanced_chunks.append({
                'text': chunk_text,
                'index': i,
                'doc_type': doc_type,
                'total_chunks': len(raw_chunks)
            })
            
        return enhanced_chunks
    
    def generate_embeddings(self, document_id: int, text: str, doc_type: str = "general") -> List[Dict[str, Any]]:
        """
        Generate embeddings for text chunks.
        
        Args:
            document_id: ID of the document
            text: Text to generate embeddings for
            doc_type: Document type for specialized chunking
            
        Returns:
            List of dictionaries with embedding information
        """
        start_time = time.time()
        
        # Use document type-specific chunking
        chunk_dicts = self.chunk_business_document(text, doc_type)
        chunks = [chunk['text'] for chunk in chunk_dicts]
        
        if not chunks:
            print(f"Warning: No chunks generated for document ID {document_id}")
            return []
            
        print(f"Generated {len(chunks)} chunks for document ID {document_id} (type: {doc_type})")
        
        embeddings = []
        batch_size = 8  # Process in batches for efficiency
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            try:
                # Generate embeddings for the batch
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                
                # Process each embedding in the batch
                for j, embedding_vector in enumerate(batch_embeddings):
                    chunk_index = i + j
                    
                    # Convert to float32 for efficient storage
                    embedding_vector = embedding_vector.astype(np.float32)
                    
                    # Create embedding entry with metadata
                    embeddings.append({
                        'document_id': document_id,
                        'chunk_text': chunks[chunk_index],
                        'embedding_vector': embedding_vector,
                        'chunk_index': chunk_index,
                        'doc_type': doc_type,
                        'total_chunks': len(chunks)
                    })
            except Exception as e:
                print(f"Error generating embeddings for batch starting at chunk {i}: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"Generated {len(embeddings)} embeddings in {elapsed_time:.2f} seconds")
        
        return embeddings
    
    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        try:
            embedding = self.model.encode(query)
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'dimension': self.embedding_dimension,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown'),
        }
        
    def embed_multiple_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple queries at once (batched).
        
        Args:
            queries: List of query strings
            
        Returns:
            List of embedding vectors
        """
        if not queries:
            return []
            
        try:
            embeddings = self.model.encode(queries, show_progress_bar=False)
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            print(f"Error generating batch query embeddings: {e}")
            # Return zero vectors as fallback
            return [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in queries]
    
    def preprocess_business_text(self, text: str) -> str:
        """
        Preprocess business text for better embedding quality.
        
        Args:
            text: Raw business text
            
        Returns:
            Preprocessed text
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Replace common business abbreviations
        abbreviations = {
            'KPI': 'Key Performance Indicator',
            'ROI': 'Return on Investment',
            'SWOT': 'Strengths Weaknesses Opportunities Threats',
            'B2B': 'Business to Business',
            'B2C': 'Business to Consumer',
            'CEO': 'Chief Executive Officer',
            'CFO': 'Chief Financial Officer',
            'CTO': 'Chief Technology Officer',
            'Q1': 'Quarter 1',
            'Q2': 'Quarter 2',
            'Q3': 'Quarter 3',
            'Q4': 'Quarter 4',
        }
        
        for abbr, full in abbreviations.items():
            # Only replace if it's a standalone abbreviation (with word boundaries)
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        return text