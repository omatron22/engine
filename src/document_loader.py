# src/document_loader.py
import os
import time
from pathlib import Path
import pandas as pd
import PyPDF2
from PyPDF2.errors import PdfReadError
import re
from typing import Dict, List, Optional, Any, Tuple
import json

class DocumentLoader:
    def __init__(self, db_instance):
        """
        Initialize the document loader.
        
        Args:
            db_instance: Database instance for storing documents
        """
        self.db = db_instance
        self.supported_extensions = {
            '.pdf': self.load_pdf,
            '.csv': self.load_csv,
            '.json': self.load_json,
            '.txt': self.load_text
        }
    
    def process_large_document(self, file_path, max_size_mb=10, chunk_size_mb=5):
        """Process unusually large documents by splitting them first."""
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        if file_size_mb <= max_size_mb:
            # Normal processing for regular-sized documents
            return self.load_pdf(file_path) if file_path.endswith('.pdf') else self.load_csv(file_path)
        
        # For large documents, split and process in chunks
        print(f"Large document detected ({file_size_mb:.2f}MB), processing in chunks")
        
        # PDF specific implementation
        if file_path.endswith('.pdf'):
            from PyPDF2 import PdfReader, PdfWriter
            reader = PdfReader(file_path)
            total_pages = len(reader.pages)
            
            # Estimate pages per chunk
            pages_per_chunk = max(1, int(total_pages * (chunk_size_mb / file_size_mb)))
            
            results = []
            for i in range(0, total_pages, pages_per_chunk):
                chunk_end = min(i + pages_per_chunk, total_pages)
                print(f"Processing PDF chunk: pages {i+1}-{chunk_end} of {total_pages}")
                
                # Create temporary PDF with just these pages
                temp_pdf = f"temp_chunk_{i}.pdf"
                writer = PdfWriter()
                
                for page_num in range(i, chunk_end):
                    writer.add_page(reader.pages[page_num])
                    
                with open(temp_pdf, 'wb') as f:
                    writer.write(f)
                
                # Process this chunk
                chunk_result = self.load_pdf(temp_pdf)
                if chunk_result:
                    results.append(chunk_result)
                
                # Clean up
                os.remove(temp_pdf)
                
            return results
        
        # CSV specific implementation
        elif file_path.endswith('.csv'):
            import pandas as pd
            
            # Determine chunk size in rows
            df = pd.read_csv(file_path, nrows=1)
            bytes_per_row = os.path.getsize(file_path) / len(pd.read_csv(file_path))
            rows_per_chunk = int((chunk_size_mb * 1024 * 1024) / bytes_per_row)
            
            results = []
            for chunk in pd.read_csv(file_path, chunksize=rows_per_chunk):
                temp_csv = f"temp_chunk.csv"
                chunk.to_csv(temp_csv, index=False)
                
                # Process this chunk
                chunk_result = self.load_csv(temp_csv)
                if chunk_result:
                    results.append(chunk_result)
                
                # Clean up
                os.remove(temp_csv)
                
            return results
            
    def load_pdf(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and process a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with document ID and content, or None if loading fails
        """
        try:
            with open(pdf_path, 'rb') as file:
                try:
                    reader = PyPDF2.PdfReader(file)
                    
                    # Check if PDF is encrypted
                    if reader.is_encrypted:
                        print(f"Warning: PDF is encrypted: {pdf_path}")
                        return None
                    
                    # Extract document info if available
                    doc_info = {}
                    if reader.metadata:
                        doc_info = {
                            'title': reader.metadata.get('/Title', ''),
                            'author': reader.metadata.get('/Author', ''),
                            'subject': reader.metadata.get('/Subject', ''),
                            'creation_date': reader.metadata.get('/CreationDate', '')
                        }
                    
                    # Extract text with page numbers
                    text = ""
                    toc = []  # Table of contents
                    
                    # First pass: look for possible headings to build TOC
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if not page_text:
                                continue
                                
                            # Find potential headings (all caps lines or numbered sections)
                            lines = page_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                # Check if line looks like a heading
                                if (line.isupper() and len(line) > 5 and len(line) < 100) or \
                                   re.match(r'^\d+(\.\d+)*\s+[A-Z]', line):
                                    toc.append({
                                        'heading': line,
                                        'page': page_num + 1
                                    })
                        except Exception as e:
                            print(f"Error extracting TOC from page {page_num+1}: {e}")
                    
                    # Second pass: extract all text
                    for page_num, page in enumerate(reader.pages):
                        try:
                            page_text = page.extract_text()
                            if page_text:
                                text += f"\n--- Page {page_num+1} ---\n{page_text}\n"
                            else:
                                print(f"Warning: Could not extract text from page {page_num+1} in {pdf_path}")
                        except Exception as e:
                            print(f"Error extracting text from page {page_num+1} in {pdf_path}: {e}")
                    
                    if not text.strip():
                        print(f"Warning: No text extracted from {pdf_path}")
                        return None
                    
                    # Add TOC to the beginning if we found any headings
                    if toc:
                        toc_text = "TABLE OF CONTENTS\n"
                        for item in toc:
                            toc_text += f"{item['heading']} (Page {item['page']})\n"
                        text = toc_text + "\n\n" + text
                    
                    # Store in database
                    title = doc_info.get('title') or Path(pdf_path).stem
                    document_type = self._determine_document_type(pdf_path, text)
                    document_id = self.db.add_document(title, text, document_type)
                    
                    return {
                        "id": document_id, 
                        "content": text, 
                        "title": title, 
                        "type": document_type,
                        "metadata": doc_info,
                        "toc": toc
                    }
                except PdfReadError as e:
                    print(f"Error reading PDF {pdf_path}: {e}")
                    return None
        except Exception as e:
            print(f"Error loading PDF {pdf_path}: {e}")
            return None
    
    def load_csv(self, csv_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and process a CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Dictionary with document ID and content, or None if loading fails
        """
        try:
            # Try to infer separator
            with open(csv_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if ',' in first_line:
                    separator = ','
                elif ';' in first_line:
                    separator = ';'
                elif '\t' in first_line:
                    separator = '\t'
                else:
                    separator = ','  # Default
            
            # Try multiple encodings if UTF-8 fails
            try:
                df = pd.read_csv(csv_path, sep=separator, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(csv_path, sep=separator, encoding='latin1')
                except:
                    df = pd.read_csv(csv_path, sep=separator, encoding='cp1252')
            
            # Check if CSV is empty
            if df.empty:
                print(f"Warning: CSV file is empty: {csv_path}")
                return None
                
            # Generate statistics about the data
            stats = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    stats[col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std())
                    }
                elif pd.api.types.is_string_dtype(df[col]):
                    # For string columns, count unique values if not too many
                    unique_values = df[col].nunique()
                    if unique_values <= 10:
                        value_counts = df[col].value_counts().to_dict()
                        stats[col] = {
                            'unique_values': unique_values,
                            'top_values': {str(k): int(v) for k, v in list(value_counts.items())[:5]}
                        }
                    else:
                        stats[col] = {
                            'unique_values': unique_values
                        }
            
            # Convert DataFrame to string representation
            content = df.to_string(index=False)
            
            # Add metadata about the table
            table_name = Path(csv_path).stem
            num_rows, num_cols = df.shape
            column_names = ', '.join(df.columns.tolist())
            
            metadata = {
                'table_name': table_name,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'columns': list(df.columns),
                'statistics': stats,
                'file_path': csv_path
            }
            
            # Determine more specific document type based on the filename
            csv_specific_type = self._determine_csv_type(csv_path)
            
            # Create enhanced content with metadata
            metadata_text = (
                f"Table Name: {table_name}\n"
                f"Document Type: {csv_specific_type}\n"
                f"Number of Rows: {num_rows}\n"
                f"Number of Columns: {num_cols}\n"
                f"Column Names: {column_names}\n\n"
            )
            
            # Add statistics summary to the content
            metadata_text += "Data Summary:\n"
            for col, col_stats in stats.items():
                metadata_text += f"- {col}: "
                if 'min' in col_stats:
                    metadata_text += f"Numeric. Range: {col_stats['min']} to {col_stats['max']}. "
                    metadata_text += f"Mean: {col_stats['mean']:.2f}. Median: {col_stats['median']:.2f}\n"
                else:
                    metadata_text += f"{col_stats.get('unique_values', 0)} unique values\n"
            
            enhanced_content = metadata_text + "\n" + content
            
            # Store in database
            title = table_name
            document_id = self.db.add_document(title, enhanced_content, csv_specific_type)
            
            return {
                "id": document_id, 
                "content": enhanced_content, 
                "title": title, 
                "type": csv_specific_type,
                "metadata": metadata,
                "dataframe": df  # Include the dataframe in case needed for further processing
            }
        except Exception as e:
            print(f"Error loading CSV {csv_path}: {e}")
            return None
    
    def load_json(self, json_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and process a JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary with document ID and content, or None if loading fails
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to formatted string
            content = json.dumps(data, indent=2)
            
            # Store in database
            title = Path(json_path).stem
            document_type = "json_data"
            document_id = self.db.add_document(title, content, document_type)
            
            return {
                "id": document_id, 
                "content": content, 
                "title": title, 
                "type": document_type,
                "data": data  # Include the parsed data in case needed for further processing
            }
        except Exception as e:
            print(f"Error loading JSON {json_path}: {e}")
            return None
    
    def load_text(self, text_path: str) -> Optional[Dict[str, Any]]:
        """
        Load and process a text file.
        
        Args:
            text_path: Path to the text file
            
        Returns:
            Dictionary with document ID and content, or None if loading fails
        """
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(text_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"Error: Could not decode {text_path} with any supported encoding")
                return None
            
            # Store in database
            title = Path(text_path).stem
            document_type = self._determine_document_type(text_path, content)
            document_id = self.db.add_document(title, content, document_type)
            
            return {
                "id": document_id, 
                "content": content, 
                "title": title, 
                "type": document_type
            }
        except Exception as e:
            print(f"Error loading text file {text_path}: {e}")
            return None
    
    def _determine_document_type(self, file_path: str, content: str = "") -> str:
        """
        Determine document type based on filename, path, and content.
        
        Args:
            file_path: Path to the file
            content: Optional document content for content-based classification
            
        Returns:
            Document type as string
        """
        path_str = str(file_path).lower()
        filename = Path(file_path).stem.lower()
        
        # First check filename patterns
        if "strategy" in filename or "strategic" in filename:
            return "strategy"
        elif "execution" in filename:
            return "execution"
        elif "market" in filename:
            return "market_assessment"
        elif "swot" in filename:
            return "swot_analysis"
        elif "risk" in filename:
            return "risk_assessment"
        elif "financial" in filename or "finance" in filename:
            return "financial"
        
        # If no match from filename, try content-based classification
        if content:
            content_lower = content.lower()
            
            # Check for strategic keywords
            strategic_keywords = [
                "strategic plan", "strategy", "vision", "mission", "objectives",
                "competitive advantage", "strategic goals", "strategic direction"
            ]
            
            execution_keywords = [
                "implementation plan", "action plan", "execution", "operational",
                "timeline", "milestones", "deliverables", "kpi", "performance indicators"
            ]
            
            market_keywords = [
                "market analysis", "customer segment", "target market", "market share",
                "market size", "market growth", "competitive landscape"
            ]
            
            swot_keywords = [
                "swot", "strengths", "weaknesses", "opportunities", "threats",
                "internal analysis", "external analysis"
            ]
            
            risk_keywords = [
                "risk assessment", "risk management", "risk mitigation", "threats",
                "contingency plan", "risk factors", "risk matrix"
            ]
            
            financial_keywords = [
                "financial", "revenue", "profit", "cost", "budget", "forecast",
                "investment", "roi", "expenses", "income statement", "balance sheet"
            ]
            
            # Count keyword occurrences
            strategic_count = sum(content_lower.count(kw) for kw in strategic_keywords)
            execution_count = sum(content_lower.count(kw) for kw in execution_keywords)
            market_count = sum(content_lower.count(kw) for kw in market_keywords)
            swot_count = sum(content_lower.count(kw) for kw in swot_keywords)
            risk_count = sum(content_lower.count(kw) for kw in risk_keywords)
            financial_count = sum(content_lower.count(kw) for kw in financial_keywords)
            
            # Find category with highest count
            counts = {
                "strategy": strategic_count,
                "execution": execution_count,
                "market_assessment": market_count,
                "swot_analysis": swot_count,
                "risk_assessment": risk_count,
                "financial": financial_count
            }
            
            max_category = max(counts, key=counts.get)
            max_count = counts[max_category]
            
            # Only use content-based classification if we have a significant number of matches
            if max_count >= 3:
                return max_category
        
        # Default to general if no specific type was determined
        return "general"
    
    def _determine_csv_type(self, csv_path: str) -> str:
        """
        Determine more specific document type for CSV files based on filename.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Specific CSV document type
        """
        filename = Path(csv_path).stem.lower()
        
        # BizGuru specific CSV tables as shown in your documentation
        if "market" in filename and any(d in filename for d in ["1", "2", "3", "4", "5"]):
            return "market_attractiveness_data"
        elif "stratpos" in filename and any(d in filename for d in ["1", "2", "3", "4", "5"]):
            return "strategic_position_data"
        elif "mybusinessspec" in filename or "compa" in filename or "compb" in filename or "compc" in filename:
            return "competition_data"
        elif "revenue" in filename or "opincome" in filename or "gm" in filename or "cfo" in filename:
            return "finance_data"
        elif "hr" in filename or "empturn" in filename or "empengage" in filename or "diversity" in filename:
            return "hr_data"
        elif "otd" in filename or "invturn" in filename or "yield" in filename or "cycletime" in filename:
            return "operations_data"
        elif "arr" in filename or "cac" in filename or "opp" in filename or "dw" in filename:
            return "sales_marketing_data"
        else:
            return "data_table"  # Generic data table type
    
    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of document dictionaries with ID and content
        """
        start_time = time.time()
        loaded_docs = []
        
        if not os.path.exists(directory_path):
            print(f"Warning: Directory does not exist: {directory_path}")
            return loaded_docs
            
        # Count total files
        total_files = 0
        supported_files = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                total_files += 1
                file_ext = os.path.splitext(file.lower())[1]
                if file_ext in self.supported_extensions:
                    supported_files += 1
                
        if total_files == 0:
            print(f"No files found in directory: {directory_path}")
            return loaded_docs
            
        print(f"Found {total_files} files in {directory_path} ({supported_files} supported)")
        
        # Process each file
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file.lower())[1]
                
                if file_ext in self.supported_extensions:
                    # Get the appropriate loader function
                    loader_func = self.supported_extensions[file_ext]
                    
                    # Load the document
                    print(f"Loading {file}...")
                    doc = loader_func(file_path)
                    
                    if doc:
                        loaded_docs.append(doc)
                        print(f"Loaded {file_ext[1:].upper()}: {file} (Type: {doc.get('type', 'unknown')})")
                    else:
                        print(f"Failed to load {file}")
                else:
                    print(f"Skipping unsupported file format: {file}")
        
        elapsed_time = time.time() - start_time
        print(f"Successfully loaded {len(loaded_docs)} out of {supported_files} supported files in {elapsed_time:.2f} seconds")
        return loaded_docs
    
    def load_and_process(self, directory_path: str, embedding_generator) -> Dict[str, Any]:
        """
        Load documents from directory and generate embeddings.
        
        Args:
            directory_path: Path to the directory containing documents
            embedding_generator: EmbeddingGenerator instance
            
        Returns:
            Stats dictionary with loading and embedding information
        """
        start_time = time.time()
        
        # Load documents
        docs = self.load_directory(directory_path)
        
        if not docs:
            return {"success": False, "message": "No documents loaded", "elapsed_time": time.time() - start_time}
        
        # Generate embeddings
        print("\nGenerating embeddings...")
        total_embeddings = 0
        
        for doc in docs:
            doc_id = doc["id"]
            doc_type = doc.get("type", "general")
            
            embeddings = embedding_generator.generate_embeddings(doc_id, doc["content"], doc_type)
            
            # Store embeddings
            for emb in embeddings:
                self.db.store_embedding(
                    emb['document_id'],
                    emb['chunk_text'],
                    emb['embedding_vector'],
                    emb['chunk_index']
                )
            
            total_embeddings += len(embeddings)
            print(f"Generated {len(embeddings)} embeddings for document: {doc.get('title', 'Unknown')}")
        
        elapsed_time = time.time() - start_time
        
        stats = {
            "success": True,
            "documents_loaded": len(docs),
            "embeddings_generated": total_embeddings,
            "elapsed_time": elapsed_time,
            "document_types": {}
        }
        
        # Count document types
        for doc in docs:
            doc_type = doc.get("type", "unknown")
            if doc_type in stats["document_types"]:
                stats["document_types"][doc_type] += 1
            else:
                stats["document_types"][doc_type] = 1
        
        return stats