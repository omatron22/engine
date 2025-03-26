#!/usr/bin/env python3
"""
QmiracTM AI-Driven Knowledge Base

A Retrieval-Augmented Generation (RAG) system for business strategy development.
"""
import os
import sys
import time
import argparse
from pathlib import Path
import signal
import atexit

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from package - use the new init functionality
from src import (
    Database, 
    DocumentLoader, 
    EmbeddingGenerator, 
    Retriever, 
    LLMManager, 
    RAGSystem,
    create_knowledge_base
)
from src.config import (
    PDF_DIR, CSV_DIR, JSON_DIR, TXT_DIR, DB_PATH, BACKUP_DIR,
    RISK_TOLERANCE_LEVELS, BUSINESS_DOMAINS
)

def setup_argparse():
    """Configure command line arguments."""
    parser = argparse.ArgumentParser(
        description="QmiracTM AI-driven Knowledge Base for business strategy development"
    )
    parser.add_argument(
        "--db-path", 
        default=DB_PATH,
        help=f"Path to database file (default: {DB_PATH})"
    )
    parser.add_argument(
        "--reload", 
        action="store_true",
        help="Force reload documents even if already in database"
    )
    parser.add_argument(
        "--backup", 
        action="store_true",
        help="Create a database backup before starting"
    )
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Optimize database before starting"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run in demo mode with sample queries"
    )
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    print("\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Display welcome message
    print("\n" + "=" * 70)
    print(f"{'QmiracTM AI-Driven Knowledge Base':^70}")
    print(f"{'Business Strategy Intelligence System':^70}")
    print("=" * 70 + "\n")
    
    try:
        # Initialize all components using the helper function
        print("Initializing system components...")
        db, document_loader, embedding_generator, retriever, llm_manager, rag_system = create_knowledge_base(args.db_path)
        
        # Register close function to ensure database is properly closed
        atexit.register(db.close)
        
        # Backup database if requested
        if args.backup:
            print("Creating database backup...")
            backup_success = db.backup_database()
            if backup_success:
                print("Backup completed successfully.")
            else:
                print("Warning: Backup failed.")
        
        # Optimize database if requested
        if args.optimize:
            print("Optimizing database...")
            db.vacuum()
        
        # Check if Ollama is running
        print("Checking LLM service...")
        if not llm_manager.check_ollama_running():
            print("Error: Ollama server is not running. Please start it with 'ollama serve'.")
            sys.exit(1)
        
        # Ensure model is available
        print("Verifying model availability...")
        if not llm_manager.ensure_model_available():
            print("Warning: Could not ensure model availability. Some features may be limited.")
        
        # Get database info
        db_info = db.get_db_info()
        print(f"\nDatabase Status: {db_info['size_mb']} MB, {db_info['tables'].get('documents', 0)} documents, {db_info['tables'].get('embeddings', 0)} embeddings")
        
        # Determine if we need to load documents
        reload_needed = args.reload
        
        if db_info['tables'].get('documents', 0) == 0:
            print("No documents found in database. Will load documents.")
            reload_needed = True
        
        # Load documents if needed
        if reload_needed:
            load_documents(document_loader, embedding_generator, db)
        
        # Run demo mode if requested
        if args.demo:
            run_demo_mode(rag_system)
        else:
            # Run interactive mode
            run_interactive_mode(rag_system, db, retriever)
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

def load_documents(document_loader, embedding_generator, db):
    """Load documents from all supported directories."""
    start_time = time.time()
    total_docs = 0
    
    # Use the new integrated loading and processing function
    for directory, name in [
        (PDF_DIR, "PDF"), 
        (CSV_DIR, "CSV"), 
        (JSON_DIR, "JSON"), 
        (TXT_DIR, "Text")
    ]:
        if any(Path(directory).glob('*')):
            print(f"\nProcessing {name} documents from {directory}...")
            stats = document_loader.load_and_process(directory, embedding_generator)
            
            if stats["success"]:
                total_docs += stats["documents_loaded"]
                print(f"Successfully processed {stats['documents_loaded']} {name} documents")
                print(f"Generated {stats['embeddings_generated']} embeddings")
                print(f"Document types: {', '.join([f'{k} ({v})' for k, v in stats['document_types'].items()])}")
            else:
                print(f"Error processing {name} documents: {stats.get('message', 'Unknown error')}")
    
    elapsed_time = time.time() - start_time
    print(f"\nDocument loading complete in {elapsed_time:.2f} seconds")
    print(f"Total documents loaded: {total_docs}")

def run_interactive_mode(rag_system, db, retriever):
    """Run interactive command-line interface."""
    print("\n" + "="*70)
    print(f"{'QmiracTM AI Assistant Ready!':^70}")
    print("="*70)
    print("Available commands:")
    print("  'exit' - Quit the application")
    print("  'strategy' - Generate a comprehensive strategy recommendation")
    print("  'search [term]' - Search for specific information")
    print("  'docs' - List all loaded documents")
    print("  'feedback' - View recent user feedback")
    print("  'help' - Show this help message")
    print("-" * 70 + "\n")
    
    while True:
        try:
            user_input = input("\n👤 Query: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  'exit' - Quit the application")
                print("  'strategy' - Generate a comprehensive strategy recommendation")
                print("  'search [term]' - Search for specific information")
                print("  'docs' - List all loaded documents")
                print("  'feedback' - View recent user feedback")
                print("  'help' - Show this help message")
                continue
                
            if user_input.lower() == 'docs':
                # Get document information
                documents = db.get_documents(limit=50)
                
                if not documents:
                    print("No documents loaded.")
                    continue
                
                print("\n📚 Loaded Documents:")
                # Group by document type
                doc_types = {}
                for doc in documents:
                    doc_type = doc['document_type']
                    if doc_type not in doc_types:
                        doc_types[doc_type] = []
                    doc_types[doc_type].append(doc)
                
                # Print documents by type
                for doc_type, docs in doc_types.items():
                    print(f"\n{doc_type.upper()} ({len(docs)}):")
                    for doc in docs:
                        print(f"  ID {doc['id']}: {doc['title']}")
                continue
                
            if user_input.lower() == 'feedback':
                # Get recent feedback
                feedback = db.get_feedback(limit=10)
                
                if not feedback:
                    print("No feedback recorded yet.")
                    continue
                
                print("\n💬 Recent Feedback:")
                for fb in feedback:
                    rating_str = "★" * fb['rating'] + "☆" * (5 - fb['rating']) if fb['rating'] else "No rating"
                    print(f"\nQuery: {fb['query']}")
                    print(f"Rating: {rating_str}")
                    if fb['feedback']:
                        print(f"Feedback: {fb['feedback']}")
                    print(f"Date: {fb['created_at']}")
                    print("-" * 40)
                continue
                
            if user_input.lower().startswith('search '):
                search_term = user_input[7:].strip()
                if not search_term:
                    print("Please provide a search term.")
                    continue
                    
                print(f"\n🔍 Searching for: {search_term}")
                results = retriever.get_relevant_documents(search_term, top_k=5)
                
                if not results:
                    print("No relevant documents found.")
                    continue
                    
                print("\nTop relevant document chunks:")
                for i, result in enumerate(results):
                    print(f"\n{i+1}. Document: {result['document_title']} (Type: {result['document_type']})")
                    print(f"   Relevance: {result['similarity']:.2f}")
                    print(f"   Content: {result['chunk_text'][:200]}...")
                continue
            
            if user_input.lower() == 'strategy':
                # Collect strategic inputs with validation
                risk_tolerance = input("\nRisk Tolerance (High/Medium/Low): ").strip()
                if risk_tolerance.capitalize() not in RISK_TOLERANCE_LEVELS:
                    print(f"Invalid risk tolerance. Using 'Medium' as default.")
                    risk_tolerance = 'Medium'
                else:
                    risk_tolerance = risk_tolerance.capitalize()
                    
                strategic_priorities = input("Strategic Priorities: ").strip()
                strategic_constraints = input("Strategic Constraints: ").strip()
                execution_priorities = input("Execution Priorities: ").strip()
                execution_constraints = input("Execution Constraints: ").strip()
                
                strategic_inputs = {
                    'risk_tolerance': risk_tolerance,
                    'strategic_priorities': strategic_priorities,
                    'strategic_constraints': strategic_constraints,
                    'execution_priorities': execution_priorities,
                    'execution_constraints': execution_constraints
                }
                
                print("\n⏳ Generating strategy recommendation...")
                start_time = time.time()
                recommendation = rag_system.generate_strategy_recommendation(strategic_inputs)
                elapsed_time = time.time() - start_time
                
                print(f"\n📊 STRATEGY RECOMMENDATION (generated in {elapsed_time:.2f} seconds)")
                print("=" * 70)
                print(recommendation)
                print("=" * 70)
                
                # Save strategy to file
                save_response = input("\nSave this strategy recommendation to file? (y/n): ").strip().lower()
                if save_response == 'y':
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"strategy_recommendation_{timestamp}.txt"
                    with open(filename, 'w') as f:
                        f.write(recommendation)
                    print(f"Strategy saved to {filename}")
                
                # Get feedback
                feedback = input("\nWould you like to provide feedback on this strategy? (y/n): ").strip().lower()
                if feedback == 'y':
                    rating = input("Rating (1-5 stars): ").strip()
                    try:
                        rating = int(rating)
                        if rating < 1 or rating > 5:
                            rating = None
                    except ValueError:
                        rating = None
                        
                    feedback_text = input("Comments (optional): ").strip()
                    if rating or feedback_text:
                        db.store_feedback("Strategy generation", recommendation, feedback_text, rating)
                        print("Thank you for your feedback!")
            else:
                # Process regular query
                print("\n⏳ Processing query...")
                start_time = time.time()
                response = rag_system.process_query(user_input)
                elapsed_time = time.time() - start_time
                
                print(f"\n🤖 Response (generated in {elapsed_time:.2f} seconds):")
                print("-" * 70)
                print(response)
                print("-" * 70)
                
                # Get feedback (optional)
                feedback_response = input("\nWas this response helpful? (y/n): ").strip().lower()
                if feedback_response == 'n':
                    feedback = input("What could be improved? (optional): ").strip()
                    rating = 2  # Below average rating
                    if feedback:
                        db.store_feedback(user_input, response, feedback, rating)
                        print("Thank you for your feedback!")
                elif feedback_response == 'y':
                    rating = input("How would you rate this response (1-5)? ").strip()
                    try:
                        rating = int(rating)
                        if 1 <= rating <= 5:
                            db.store_feedback(user_input, response, "Helpful", rating)
                            print("Thank you for your feedback!")
                    except ValueError:
                        pass
                
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError processing query: {e}")
    
    print("\nQmiracTM AI Assistant shutting down.")

def run_demo_mode(rag_system):
    """Run a demonstration with sample queries."""
    demo_queries = [
        "What factors should I consider in market attractiveness assessment?",
        "How can I improve my business's strategic position?",
        "What are the key components of a SWOT analysis?",
        "How should I formulate my execution priorities?",
        "What's the relationship between risk tolerance and strategic planning?"
    ]
    
    print("\n" + "="*70)
    print(f"{'QmiracTM AI Assistant Demo Mode':^70}")
    print("="*70)
    print("Running demonstration with sample business strategy queries.")
    print("Press Ctrl+C at any time to exit the demo.")
    
    try:
        for i, query in enumerate(demo_queries):
            print(f"\n\nDemo Query {i+1}/{len(demo_queries)}:")
            print(f"👤 {query}")
            print("\n⏳ Processing...")
            
            start_time = time.time()
            response = rag_system.process_query(query)
            elapsed_time = time.time() - start_time
            
            print(f"\n🤖 Response (generated in {elapsed_time:.2f} seconds):")
            print("-" * 70)
            print(response)
            print("-" * 70)
            
            # Pause between queries
            if i < len(demo_queries) - 1:
                input("\nPress Enter to continue to the next query...")
    
        # Demo strategy generation
        print("\n\n" + "="*70)
        print(f"{'Demo Strategy Generation':^70}")
        print("="*70)
        
        print("\nGenerating a sample business strategy with default inputs...")
        strategic_inputs = {
            'risk_tolerance': 'Medium',
            'strategic_priorities': 'Market expansion, product innovation, customer retention',
            'strategic_constraints': 'Limited capital, competitive market',
            'execution_priorities': 'Sales growth, operational efficiency',
            'execution_constraints': 'Resource limitations, regulatory requirements'
        }
        
        print("\n⏳ Generating strategy recommendation...")
        start_time = time.time()
        recommendation = rag_system.generate_strategy_recommendation(strategic_inputs)
        elapsed_time = time.time() - start_time
        
        print(f"\n📊 STRATEGY RECOMMENDATION (generated in {elapsed_time:.2f} seconds)")
        print("=" * 70)
        print(recommendation)
        print("=" * 70)
        
        print("\nDemo completed! Switching to interactive mode...\n")
        run_interactive_mode(rag_system, None, None)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError in demo mode: {e}")

if __name__ == "__main__":
    main()