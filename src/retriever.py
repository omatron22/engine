# src/retriever.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple
from src.config import TOP_K_RESULTS
import time

class Retriever:
    def __init__(self, db_instance, embedding_generator):
        """
        Initialize the retriever with database and embedding generator.
        
        Args:
            db_instance: Database instance for retrieving embeddings
            embedding_generator: EmbeddingGenerator for query encoding
        """
        self.db = db_instance
        self.embedding_generator = embedding_generator
        
    def get_relevant_documents(self, query: str, top_k: int = TOP_K_RESULTS, 
                              filter_type: Optional[str] = None,
                              similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant document chunks for the query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            filter_type: Optional document type filter
            similarity_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant document chunks with metadata
        """
        start_time = time.time()
        
        # Normalize the query
        query = query.strip()
        if not query:
            return []
            
        # Generate embedding for the query
        query_embedding = self.embedding_generator.get_query_embedding(query)
        
        # Get all embeddings from the database
        all_embeddings = self.db.get_embeddings()
        
        if not all_embeddings:
            print("No embeddings found in the database")
            return []
        
        print(f"Searching among {len(all_embeddings)} document chunks")
        
        # Apply document type filter if specified
        if filter_type:
            filtered_embeddings = []
            for emb in all_embeddings:
                doc_id = emb['document_id']
                doc = self.db.get_document_by_id(doc_id)
                if doc and doc[3] == filter_type:  # Index 3 is document_type
                    filtered_embeddings.append(emb)
                    
            all_embeddings = filtered_embeddings
            print(f"Filtered to {len(all_embeddings)} chunks of type '{filter_type}'")
            
            if not all_embeddings:
                return []
        
        # Extract embedding vectors
        embedding_vectors = [item['embedding_vector'] for item in all_embeddings]
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], embedding_vectors)[0]
        
        # Apply similarity threshold
        above_threshold_indices = [i for i, score in enumerate(similarities) if score >= similarity_threshold]
        
        if not above_threshold_indices:
            print(f"No document chunks with similarity above threshold {similarity_threshold}")
            # Return top matches anyway if nothing above threshold
            top_indices = np.argsort(similarities)[-min(top_k, len(similarities)):][::-1]
        else:
            # Get indices of top-k most similar chunks above threshold
            filtered_similarities = [(i, similarities[i]) for i in above_threshold_indices]
            sorted_indices = sorted(filtered_similarities, key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in sorted_indices[:top_k]]
        
        # Get the actual chunks with their similarity scores and metadata
        results = []
        for idx in top_indices:
            chunk_data = all_embeddings[idx]
            document_id = chunk_data['document_id']
            document = self.db.get_document_by_id(document_id)
            
            if document:
                document_title = document[1]  # Index 1 is title
                document_type = document[3]  # Index 3 is document_type
            else:
                document_title = "Unknown"
                document_type = "Unknown"
                
            # Add the current chunk to results
            results.append({
                'chunk_text': chunk_data['chunk_text'],
                'document_id': document_id,
                'document_title': document_title,
                'document_type': document_type,
                'chunk_index': chunk_data['chunk_index'],
                'similarity': similarities[idx]
            })
            
            # For context, also get surrounding chunks if they're from the same document
            if chunk_data['chunk_index'] > 0:  # If not the first chunk
                self._add_surrounding_context(results, all_embeddings, document_id, chunk_data['chunk_index'] - 1)
            
            # Get the next chunk too for context
            self._add_surrounding_context(results, all_embeddings, document_id, chunk_data['chunk_index'] + 1)
        
        # Remove duplicates that might have been added from surrounding context
        results = self._deduplicate_results(results)
        
        # Re-sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Trim to the requested number
        results = results[:top_k]
        
        retrieval_time = time.time() - start_time
        print(f"Found {len(results)} relevant document chunks in {retrieval_time:.2f} seconds")
        return results
    
    def _add_surrounding_context(self, results: List[Dict], all_embeddings: List[Dict], 
                                document_id: int, chunk_index: int) -> None:
        """Add a surrounding chunk for context if it exists."""
        for emb in all_embeddings:
            if (emb['document_id'] == document_id and 
                emb['chunk_index'] == chunk_index):
                
                # Add with lower similarity to ensure it ranks below direct matches
                document = self.db.get_document_by_id(document_id)
                if document:
                    document_title = document[1]
                    document_type = document[3]
                else:
                    document_title = "Unknown"
                    document_type = "Unknown"
                    
                results.append({
                    'chunk_text': emb['chunk_text'],
                    'document_id': document_id,
                    'document_title': document_title,
                    'document_type': document_type,
                    'chunk_index': chunk_index,
                    'similarity': 0.1  # Lower similarity for context
                })
                break
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on document_id and chunk_index."""
        seen = set()
        deduped = []
        
        for result in results:
            key = (result['document_id'], result['chunk_index'])
            if key not in seen:
                seen.add(key)
                deduped.append(result)
                
        return deduped
    
    def get_document_by_title(self, title: str) -> Optional[Dict]:
        """
        Retrieve a document by its title.
        
        Args:
            title: Document title to search for
            
        Returns:
            Document dictionary or None if not found
        """
        cursor = self.db.conn.cursor()
        query = "SELECT * FROM documents WHERE title = ?"
        cursor.execute(query, (title,))
        doc = cursor.fetchone()
        
        if doc:
            return {
                'id': doc[0],
                'title': doc[1],
                'content': doc[2],
                'document_type': doc[3],
                'created_at': doc[4]
            }
        return None
        
    def retrieve_by_document_type(self, doc_type: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve documents by type.
        
        Args:
            doc_type: Document type to filter by
            top_k: Maximum number of documents to return
            
        Returns:
            List of document dictionaries
        """
        cursor = self.db.conn.cursor()
        query = "SELECT * FROM documents WHERE document_type = ? LIMIT ?"
        cursor.execute(query, (doc_type, top_k))
        docs = cursor.fetchall()
        
        results = []
        for doc in docs:
            results.append({
                'id': doc[0],
                'title': doc[1],
                'content': doc[2],
                'document_type': doc[3],
                'created_at': doc[4]
            })
        
        return results
    
    def search_documents(self, search_term: str, 
                        doc_types: Optional[List[str]] = None) -> List[Dict]:
        """
        Search for documents containing the search term.
        
        Args:
            search_term: Term to search for in documents
            doc_types: Optional list of document types to filter by
            
        Returns:
            List of matching documents
        """
        cursor = self.db.conn.cursor()
        
        if doc_types:
            placeholders = ','.join(['?' for _ in doc_types])
            query = f"""
            SELECT * FROM documents 
            WHERE (title LIKE ? OR content LIKE ?) 
            AND document_type IN ({placeholders})
            ORDER BY created_at DESC
            """
            params = [f'%{search_term}%', f'%{search_term}%'] + doc_types
        else:
            query = """
            SELECT * FROM documents 
            WHERE title LIKE ? OR content LIKE ?
            ORDER BY created_at DESC
            """
            params = [f'%{search_term}%', f'%{search_term}%']
            
        cursor.execute(query, params)
        docs = cursor.fetchall()
        
        results = []
        for doc in docs:
            results.append({
                'id': doc[0],
                'title': doc[1],
                'content': doc[2],
                'document_type': doc[3],
                'created_at': doc[4]
            })
        
        return results
        
    def get_similar_documents(self, document_id: int, 
                             top_k: int = 5) -> List[Dict]:
        """
        Find documents similar to the specified document.
        
        Args:
            document_id: ID of the document to find similar documents for
            top_k: Number of similar documents to return
            
        Returns:
            List of similar documents with similarity scores
        """
        document = self.db.get_document_by_id(document_id)
        if not document:
            print(f"Document with ID {document_id} not found")
            return []
            
        # Get document embeddings
        cursor = self.db.conn.cursor()
        query = "SELECT embedding_vector FROM embeddings WHERE document_id = ? LIMIT 1"
        cursor.execute(query, (document_id,))
        result = cursor.fetchone()
        
        if not result:
            print(f"No embeddings found for document ID {document_id}")
            return []
            
        # Use the document's embedding to find similar documents
        doc_embedding = np.frombuffer(result[0], dtype=np.float32)
        
        # Get all document embeddings
        all_docs = {}
        cursor.execute("SELECT DISTINCT document_id FROM embeddings")
        for row in cursor.fetchall():
            all_docs[row[0]] = True
            
        similar_docs = []
        for doc_id in all_docs:
            if doc_id == document_id:
                continue
                
            similarity = self._calculate_document_similarity(document_id, doc_id)
            if similarity > 0:
                doc = self.db.get_document_by_id(doc_id)
                similar_docs.append({
                    'id': doc_id,
                    'title': doc[1],
                    'document_type': doc[3],
                    'similarity': similarity
                })
                
        # Sort by similarity and return top_k
        similar_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_docs[:top_k]
        
    def _calculate_document_similarity(self, doc_id1: int, doc_id2: int) -> float:
        """Calculate average similarity between chunks of two documents."""
        cursor = self.db.conn.cursor()
        
        # Get embeddings for first document
        cursor.execute("SELECT embedding_vector FROM embeddings WHERE document_id = ?", (doc_id1,))
        emb1_results = cursor.fetchall()
        
        # Get embeddings for second document
        cursor.execute("SELECT embedding_vector FROM embeddings WHERE document_id = ?", (doc_id2,))
        emb2_results = cursor.fetchall()
        
        if not emb1_results or not emb2_results:
            return 0.0
            
        # Convert binary blobs to numpy arrays
        emb1_vectors = [np.frombuffer(emb[0], dtype=np.float32) for emb in emb1_results]
        emb2_vectors = [np.frombuffer(emb[0], dtype=np.float32) for emb in emb2_results]
        
        # Calculate pairwise similarities
        similarities = []
        for v1 in emb1_vectors:
            for v2 in emb2_vectors:
                sim = cosine_similarity([v1], [v2])[0][0]
                similarities.append(sim)
                
        # Return average similarity
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.0