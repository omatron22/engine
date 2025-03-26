# src/db.py
import sqlite3
import os
import time
from pathlib import Path
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import json

class Database:
    def __init__(self, db_path='database/kb.db'):
        """
        Initialize the database for the QmiracTM AI Knowledge Base.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = self._create_connection()
        
        # Enable foreign keys
        cursor = self.conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Initialize database schema
        self._initialize_db()
        
        # Set up pragmas for better performance
        self._optimize_db()
        
        # Cache frequently used queries
        self.prepared_statements = {}
        
    def _create_connection(self) -> sqlite3.Connection:
        """Create and return a database connection."""
        return sqlite3.connect(self.db_path)
    
    def _optimize_db(self) -> None:
        """Set SQLite pragmas for better performance."""
        cursor = self.conn.cursor()
        # Use WAL mode for better concurrency
        cursor.execute("PRAGMA journal_mode = WAL")
        # Use memory for temp storage
        cursor.execute("PRAGMA temp_store = MEMORY")
        # Larger cache size for better performance
        cursor.execute("PRAGMA cache_size = -10000")  # ~10MB cache
        # Sync less often for better performance (slight durability trade-off)
        cursor.execute("PRAGMA synchronous = NORMAL")
        # Set busy timeout to prevent "database is locked" errors
        cursor.execute("PRAGMA busy_timeout = 5000")  # 5 seconds
        
        self.conn.commit()
    
    def _initialize_db(self) -> None:
        """Initialize the database with schema."""
        cursor = self.conn.cursor()
        schema_path = Path('database/schema.sql')
        with open(schema_path, 'r') as f:
            schema_script = f.read()
        cursor.executescript(schema_script)
        self.conn.commit()
        
        # Check if tables were created
        tables = self._get_tables()
        expected_tables = ['documents', 'embeddings', 'strategic_inputs', 'user_feedback']
        
        for table in expected_tables:
            if table not in tables:
                print(f"Warning: Table '{table}' was not created properly")
    
    def _get_tables(self) -> List[str]:
        """Get a list of tables in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cursor.fetchall()]
    
    def get_db_info(self) -> Dict[str, Any]:
        """Get information about the database."""
        cursor = self.conn.cursor()
        
        # Get table counts
        table_counts = {}
        for table in self._get_tables():
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_counts[table] = cursor.fetchone()[0]
        
        # Get database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        
        # Get document type counts
        cursor.execute("SELECT document_type, COUNT(*) FROM documents GROUP BY document_type")
        doc_type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "db_path": self.db_path,
            "tables": table_counts,
            "document_types": doc_type_counts,
            "size_bytes": db_size,
            "size_mb": round(db_size / (1024 * 1024), 2)
        }
    
    def add_document(self, title: str, content: str, document_type: str) -> int:
        """
        Add a document to the database.
        
        Args:
            title: Document title
            content: Document content
            document_type: Type of document
            
        Returns:
            Document ID
        """
        # Check if document with same title already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE title = ?", (title,))
        existing = cursor.fetchone()
        
        if existing:
            print(f"Warning: Document with title '{title}' already exists (id: {existing[0]})")
            # You could choose to update or skip here
            # For now, we'll continue adding as a new document
        
        query = """
        INSERT INTO documents (title, content, document_type)
        VALUES (?, ?, ?)
        """
        cursor.execute(query, (title, content, document_type))
        self.conn.commit()
        
        doc_id = cursor.lastrowid
        print(f"Added document: '{title}' (id: {doc_id}, type: {document_type})")
        
        return doc_id
    
    def update_document(self, document_id: int, title: Optional[str] = None, 
                       content: Optional[str] = None, document_type: Optional[str] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            document_id: ID of document to update
            title: New title (if None, not updated)
            content: New content (if None, not updated)
            document_type: New document type (if None, not updated)
            
        Returns:
            True if update successful, False otherwise
        """
        # Check if document exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE id = ?", (document_id,))
        if not cursor.fetchone():
            print(f"Error: Document with ID {document_id} not found")
            return False
        
        # Build update query dynamically based on provided fields
        update_fields = []
        params = []
        
        if title is not None:
            update_fields.append("title = ?")
            params.append(title)
        
        if content is not None:
            update_fields.append("content = ?")
            params.append(content)
        
        if document_type is not None:
            update_fields.append("document_type = ?")
            params.append(document_type)
        
        if not update_fields:
            print("Warning: No fields to update")
            return False
        
        # Add document_id to params
        params.append(document_id)
        
        query = f"""
        UPDATE documents 
        SET {', '.join(update_fields)}
        WHERE id = ?
        """
        
        cursor.execute(query, params)
        self.conn.commit()
        
        # Check if update was successful
        return cursor.rowcount > 0
    
    def delete_document(self, document_id: int, delete_embeddings: bool = True) -> bool:
        """
        Delete a document and optionally its embeddings.
        
        Args:
            document_id: ID of document to delete
            delete_embeddings: Whether to also delete associated embeddings
            
        Returns:
            True if delete successful, False otherwise
        """
        cursor = self.conn.cursor()
        
        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            # Delete embeddings if requested
            if delete_embeddings:
                cursor.execute("DELETE FROM embeddings WHERE document_id = ?", (document_id,))
                embeddings_deleted = cursor.rowcount
                print(f"Deleted {embeddings_deleted} embeddings for document ID {document_id}")
            
            # Delete document
            cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            doc_deleted = cursor.rowcount > 0
            
            # Commit transaction
            self.conn.commit()
            
            if doc_deleted:
                print(f"Successfully deleted document ID {document_id}")
            else:
                print(f"Document ID {document_id} not found")
                
            return doc_deleted
            
        except Exception as e:
            # Rollback transaction on error
            self.conn.rollback()
            print(f"Error deleting document: {e}")
            return False
    
    def store_embedding(self, document_id: int, chunk_text: str, 
                       embedding_vector: np.ndarray, chunk_index: int) -> int:
        """
        Store an embedding in the database.
        
        Args:
            document_id: ID of the associated document
            chunk_text: Text chunk that was embedded
            embedding_vector: Numpy array containing the embedding
            chunk_index: Index of this chunk in the document
            
        Returns:
            Embedding ID
        """
        # Verify document exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE id = ?", (document_id,))
        if not cursor.fetchone():
            print(f"Error: Document ID {document_id} not found. Embedding not stored.")
            return -1
        
        # Convert numpy array to binary blob
        embedding_blob = embedding_vector.tobytes()
        
        # Check if an embedding for this chunk already exists
        cursor.execute(
            "SELECT id FROM embeddings WHERE document_id = ? AND chunk_index = ?", 
            (document_id, chunk_index)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing embedding
            query = """
            UPDATE embeddings 
            SET chunk_text = ?, embedding_vector = ?
            WHERE id = ?
            """
            cursor.execute(query, (chunk_text, embedding_blob, existing[0]))
            self.conn.commit()
            return existing[0]
        else:
            # Insert new embedding
            query = """
            INSERT INTO embeddings (document_id, chunk_text, embedding_vector, chunk_index)
            VALUES (?, ?, ?, ?)
            """
            cursor.execute(query, (document_id, chunk_text, embedding_blob, chunk_index))
            self.conn.commit()
            return cursor.lastrowid
    
    def batch_store_embeddings(self, embeddings: List[Dict[str, Any]]) -> int:
        """
        Store multiple embeddings in a single transaction.
        
        Args:
            embeddings: List of embedding dictionaries
            
        Returns:
            Number of embeddings stored
        """
        if not embeddings:
            return 0
            
        cursor = self.conn.cursor()
        
        try:
            # Begin transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            query = """
            INSERT INTO embeddings (document_id, chunk_text, embedding_vector, chunk_index)
            VALUES (?, ?, ?, ?)
            """
            
            count = 0
            for emb in embeddings:
                document_id = emb['document_id']
                chunk_text = emb['chunk_text']
                embedding_vector = emb['embedding_vector']
                chunk_index = emb['chunk_index']
                
                # Convert numpy array to binary blob
                embedding_blob = embedding_vector.tobytes()
                
                cursor.execute(query, (document_id, chunk_text, embedding_blob, chunk_index))
                count += 1
            
            # Commit transaction
            self.conn.commit()
            
            return count
            
        except Exception as e:
            # Rollback transaction on error
            self.conn.rollback()
            print(f"Error batch storing embeddings: {e}")
            return 0
    
    def get_embeddings(self, document_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get embeddings from the database, optionally filtered by document ID.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            List of embedding dictionaries
        """
        cursor = self.conn.cursor()
        
        if document_id is not None:
            query = """
            SELECT id, document_id, chunk_text, embedding_vector, chunk_index
            FROM embeddings
            WHERE document_id = ?
            """
            cursor.execute(query, (document_id,))
        else:
            query = """
            SELECT id, document_id, chunk_text, embedding_vector, chunk_index
            FROM embeddings
            """
            cursor.execute(query)
        
        results = []
        for row in cursor.fetchall():
            embedding_bytes = row[3]
            # Convert binary blob back to numpy array
            embedding_vector = np.frombuffer(embedding_bytes, dtype=np.float32)
            results.append({
                'id': row[0],
                'document_id': row[1],
                'chunk_text': row[2],
                'embedding_vector': embedding_vector,
                'chunk_index': row[4]
            })
        
        return results
    
    def get_embeddings_count(self, document_id: Optional[int] = None) -> int:
        """
        Get count of embeddings, optionally filtered by document ID.
        
        Args:
            document_id: Optional document ID to filter by
            
        Returns:
            Count of embeddings
        """
        cursor = self.conn.cursor()
        
        if document_id is not None:
            query = "SELECT COUNT(*) FROM embeddings WHERE document_id = ?"
            cursor.execute(query, (document_id,))
        else:
            query = "SELECT COUNT(*) FROM embeddings"
            cursor.execute(query)
        
        return cursor.fetchone()[0]
    
    def store_strategic_input(self, risk_tolerance: str, strategic_priorities: str, 
                             strategic_constraints: str, execution_priorities: str,
                             execution_constraints: str) -> int:
        """
        Store strategic inputs for business strategy generation.
        
        Args:
            risk_tolerance: Risk tolerance level (High/Medium/Low)
            strategic_priorities: Strategic priorities text
            strategic_constraints: Strategic constraints text
            execution_priorities: Execution priorities text
            execution_constraints: Execution constraints text
            
        Returns:
            Strategic input ID
        """
        # Validate risk tolerance
        valid_risk_levels = ['High', 'Medium', 'Low']
        if risk_tolerance not in valid_risk_levels:
            print(f"Warning: Invalid risk tolerance '{risk_tolerance}'. Using 'Medium'.")
            risk_tolerance = 'Medium'
        
        cursor = self.conn.cursor()
        query = """
        INSERT INTO strategic_inputs (
            risk_tolerance, strategic_priorities, strategic_constraints,
            execution_priorities, execution_constraints
        ) VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            risk_tolerance, strategic_priorities, strategic_constraints,
            execution_priorities, execution_constraints
        ))
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_strategic_inputs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent strategic inputs.
        
        Args:
            limit: Maximum number of inputs to return
            
        Returns:
            List of strategic input dictionaries
        """
        cursor = self.conn.cursor()
        query = """
        SELECT id, risk_tolerance, strategic_priorities, strategic_constraints,
               execution_priorities, execution_constraints, created_at
        FROM strategic_inputs
        ORDER BY created_at DESC
        LIMIT ?
        """
        cursor.execute(query, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'risk_tolerance': row[1],
                'strategic_priorities': row[2],
                'strategic_constraints': row[3],
                'execution_priorities': row[4],
                'execution_constraints': row[5],
                'created_at': row[6]
            })
        
        return results
    
    def store_feedback(self, query: str, response: str, 
                      feedback: Optional[str] = None, 
                      rating: Optional[int] = None) -> int:
        """
        Store user feedback for a query-response pair.
        
        Args:
            query: The user's query
            response: The system's response
            feedback: Optional user feedback text
            rating: Optional rating (1-5)
            
        Returns:
            Feedback ID
        """
        # Validate rating if provided
        if rating is not None and (rating < 1 or rating > 5):
            print(f"Warning: Invalid rating {rating}. Rating should be between 1 and 5.")
            rating = None
        
        cursor = self.conn.cursor()
        query_sql = """
        INSERT INTO user_feedback (query, response, feedback, rating)
        VALUES (?, ?, ?, ?)
        """
        cursor.execute(query_sql, (query, response, feedback, rating))
        self.conn.commit()
        
        return cursor.lastrowid
    
    def get_feedback(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recent user feedback.
        
        Args:
            limit: Maximum number of feedback entries to return
            
        Returns:
            List of feedback dictionaries
        """
        cursor = self.conn.cursor()
        query = """
        SELECT id, query, response, feedback, rating, created_at
        FROM user_feedback
        ORDER BY created_at DESC
        LIMIT ?
        """
        cursor.execute(query, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'query': row[1],
                'response': row[2],
                'feedback': row[3],
                'rating': row[4],
                'created_at': row[5]
            })
        
        return results
    
    def get_document_by_id(self, document_id: int) -> Optional[Tuple]:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document ID to retrieve
            
        Returns:
            Document tuple or None if not found
        """
        cursor = self.conn.cursor()
        query = "SELECT * FROM documents WHERE id = ?"
        cursor.execute(query, (document_id,))
        return cursor.fetchone()
    
    def get_documents(self, document_type: Optional[str] = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get documents, optionally filtered by type.
        
        Args:
            document_type: Optional document type to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of document dictionaries
        """
        cursor = self.conn.cursor()
        
        if document_type:
            query = """
            SELECT id, title, document_type, created_at
            FROM documents
            WHERE document_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
            cursor.execute(query, (document_type, limit))
        else:
            query = """
            SELECT id, title, document_type, created_at
            FROM documents
            ORDER BY created_at DESC
            LIMIT ?
            """
            cursor.execute(query, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1],
                'document_type': row[2],
                'created_at': row[3]
            })
        
        return results
    
    def search_documents(self, search_term: str, 
                        document_type: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search for documents containing the search term.
        
        Args:
            search_term: Term to search for
            document_type: Optional document type to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of matching document dictionaries
        """
        cursor = self.conn.cursor()
        
        if document_type:
            query = """
            SELECT id, title, document_type, created_at
            FROM documents
            WHERE (title LIKE ? OR content LIKE ?) AND document_type = ?
            ORDER BY created_at DESC
            LIMIT ?
            """
            cursor.execute(query, (f'%{search_term}%', f'%{search_term}%', document_type, limit))
        else:
            query = """
            SELECT id, title, document_type, created_at
            FROM documents
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """
            cursor.execute(query, (f'%{search_term}%', f'%{search_term}%', limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row[0],
                'title': row[1],
                'document_type': row[2],
                'created_at': row[3]
            })
        
        return results
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file, or auto-generated if None
            
        Returns:
            True if backup successful, False otherwise
        """
        if not backup_path:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = f"{os.path.splitext(self.db_path)[0]}_{timestamp}.backup.db"
        
        try:
            # Create a new connection to the backup file
            backup_conn = sqlite3.connect(backup_path)
            
            # Copy the database to the backup file
            self.conn.backup(backup_conn)
            
            # Close the backup connection
            backup_conn.close()
            
            print(f"Database backup created at: {backup_path}")
            return True
        except Exception as e:
            print(f"Error creating database backup: {e}")
            return False
    
    def vacuum(self) -> bool:
        """
        Optimize the database by rebuilding it to reclaim unused space.
        
        Returns:
            True if vacuum successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("VACUUM")
            self.conn.commit()
            print("Database optimized")
            return True
        except Exception as e:
            print(f"Error optimizing database: {e}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed")