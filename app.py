#!/usr/bin/env python3
"""
QmiracTM AI-Driven Knowledge Base Demo

A polished demo for the Retrieval-Augmented Generation (RAG) system 
for offline business strategy development.
"""
import os
import sys
import time
import argparse
import signal
import atexit
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
from rich.prompt import Prompt, Confirm
import shutil

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from package
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
from src.output_generator import StrategyOutputGenerator

# Initialize rich console for better UI
console = Console()

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
    console.print("\n[bold red]Received interrupt signal. Shutting down gracefully...[/bold red]")
    sys.exit(0)

def display_welcome_banner():
    """Display a nicely formatted welcome banner."""
    title = "QmiracTM AI-Driven Knowledge Base"
    subtitle = "Business Strategy Intelligence System"
    version = "v0.1.0"
    
    width = shutil.get_terminal_size().columns
    width = min(width, 100)  # Cap at 100 columns
    
    console.print("\n")
    console.print(Panel(
        f"[bold blue]{title}[/bold blue]\n[yellow]{subtitle}[/yellow]\n[dim]{version}[/dim]",
        width=width,
        border_style="blue",
        expand=False
    ))

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = setup_argparse()
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Display welcome message
    display_welcome_banner()
    
    try:
        with Progress() as progress:
            init_task = progress.add_task("[cyan]Initializing system components...", total=100)
            
            # Set progress to 20%
            progress.update(init_task, completed=20)
            
            # Initialize all components using the helper function
            db, document_loader, embedding_generator, retriever, llm_manager, rag_system = create_knowledge_base(args.db_path)
            
            # Initialize PDF output generator
            strategy_output_generator = StrategyOutputGenerator()
            
            # Register close function to ensure database is properly closed
            atexit.register(db.close)
            
            # Set progress to 40%
            progress.update(init_task, completed=40)
            
            # Backup database if requested
            if args.backup:
                progress.update(init_task, description="[cyan]Creating database backup...")
                backup_success = db.backup_database()
                if backup_success:
                    progress.update(init_task, description="[green]Backup completed successfully.")
                else:
                    progress.update(init_task, description="[red]Warning: Backup failed.")
                time.sleep(0.5)  # Short pause to show message
            
            # Set progress to 60%
            progress.update(init_task, completed=60)
            
            # Optimize database if requested
            if args.optimize:
                progress.update(init_task, description="[cyan]Optimizing database...")
                db.vacuum()
                time.sleep(0.5)  # Short pause to show message
            
            # Set progress to 80%
            progress.update(init_task, completed=80)
            
            # Check if Ollama is running
            progress.update(init_task, description="[cyan]Checking LLM service...")
            if not llm_manager.check_ollama_running():
                progress.update(init_task, completed=100)
                console.print("\n[bold red]Error:[/bold red] Ollama server is not running. Please start it with 'ollama serve'.")
                sys.exit(1)
            
            # Ensure model is available
            progress.update(init_task, description="[cyan]Verifying model availability...")
            if not llm_manager.ensure_model_available():
                console.print("[yellow]Warning:[/yellow] Could not ensure model availability. Some features may be limited.")
            
            # Complete the progress bar
            progress.update(init_task, completed=100)
        
        # Get database info
        db_info = db.get_db_info()
        
        # Create a nice table for database info
        table = Table(title="Database Status", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Size", f"{db_info['size_mb']} MB")
        table.add_row("Documents", f"{db_info['tables'].get('documents', 0)}")
        table.add_row("Embeddings", f"{db_info['tables'].get('embeddings', 0)}")
        
        # Add document types if available
        if 'document_types' in db_info and db_info['document_types']:
            for doc_type, count in db_info['document_types'].items():
                table.add_row(f"  {doc_type}", f"{count}")
        
        console.print(table)
        
        # Determine if we need to load documents
        reload_needed = args.reload
        
        if db_info['tables'].get('documents', 0) == 0:
            console.print("\n[yellow]No documents found in database. Will load documents.[/yellow]")
            reload_needed = True
        
        # Load documents if needed
        if reload_needed:
            load_documents(document_loader, embedding_generator, db)
        
        # Run demo mode if requested
        if args.demo:
            run_demo_mode(rag_system, strategy_output_generator)
        else:
            # Run interactive mode
            run_interactive_mode(rag_system, db, retriever, strategy_output_generator)
        
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

def load_documents(document_loader, embedding_generator, db):
    """Load documents from all supported directories."""
    start_time = time.time()
    total_docs = 0
    
    console.print("\n[bold blue]Loading Documents[/bold blue]")
    
    # Use the integrated loading and processing function
    for directory, name in [
        (PDF_DIR, "PDF"), 
        (CSV_DIR, "CSV"), 
        (JSON_DIR, "JSON"), 
        (TXT_DIR, "Text")
    ]:
        if any(Path(directory).glob('*')):
            console.print(f"\n[cyan]Processing {name} documents from {directory}...[/cyan]")
            
            with Progress() as progress:
                task = progress.add_task(f"[green]Loading {name} documents...", total=100)
                
                # Process halfway indication
                progress.update(task, completed=50)
                
                stats = document_loader.load_and_process(directory, embedding_generator)
                
                # Complete the task
                progress.update(task, completed=100)
            
            if stats["success"]:
                total_docs += stats["documents_loaded"]
                console.print(f"✅ Successfully processed [bold green]{stats['documents_loaded']}[/bold green] {name} documents")
                console.print(f"   Generated [bold green]{stats['embeddings_generated']}[/bold green] embeddings")
                
                # Show document types
                if stats['document_types']:
                    types_table = Table(show_header=True, header_style="dim")
                    types_table.add_column("Document Type", style="cyan")
                    types_table.add_column("Count", style="green", justify="right")
                    
                    for k, v in stats['document_types'].items():
                        types_table.add_row(k, str(v))
                    
                    console.print(types_table)
            else:
                console.print(f"❌ [bold red]Error processing {name} documents:[/bold red] {stats.get('message', 'Unknown error')}")
    
    elapsed_time = time.time() - start_time
    console.print(f"\n[bold green]Document loading complete in {elapsed_time:.2f} seconds[/bold green]")
    console.print(f"[bold blue]Total documents loaded: {total_docs}[/bold blue]")

def run_interactive_mode(rag_system, db, retriever, strategy_output_generator):
    """Run interactive command-line interface."""
    console.print("\n" + "=" * shutil.get_terminal_size().columns)
    console.print("[bold blue]QmiracTM AI Assistant Ready![/bold blue]", justify="center")
    console.print("=" * shutil.get_terminal_size().columns)
    
    help_table = Table(show_header=True, header_style="bold cyan")
    help_table.add_column("Command", style="yellow")
    help_table.add_column("Description", style="white")
    
    help_table.add_row("exit", "Quit the application")
    help_table.add_row("strategy", "Generate a comprehensive strategy recommendation")
    help_table.add_row("search [term]", "Search for specific information")
    help_table.add_row("docs", "List all loaded documents")
    help_table.add_row("feedback", "View recent user feedback")
    help_table.add_row("help", "Show this help message")
    
    console.print(help_table)
    console.print("-" * shutil.get_terminal_size().columns + "\n")
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold cyan]👤 Query[/bold cyan]")
            
            if not user_input:
                continue
                
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower() == 'help':
                console.print(help_table)
                continue
                
            if user_input.lower() == 'docs':
                # Get document information
                documents = db.get_documents(limit=50)
                
                if not documents:
                    console.print("[yellow]No documents loaded.[/yellow]")
                    continue
                
                console.print("\n[bold blue]📚 Loaded Documents:[/bold blue]")
                
                # Group by document type
                doc_types = {}
                for doc in documents:
                    doc_type = doc['document_type']
                    if doc_type not in doc_types:
                        doc_types[doc_type] = []
                    doc_types[doc_type].append(doc)
                
                # Print documents by type
                for doc_type, docs in doc_types.items():
                    console.print(f"\n[bold cyan]{doc_type.upper()} ({len(docs)}):[/bold cyan]")
                    
                    docs_table = Table(show_header=False, box=None)
                    docs_table.add_column("ID", style="dim")
                    docs_table.add_column("Title", style="green")
                    
                    for doc in docs:
                        docs_table.add_row(f"ID {doc['id']}", doc['title'])
                    
                    console.print(docs_table)
                continue
                
            if user_input.lower() == 'feedback':
                # Get recent feedback
                feedback = db.get_feedback(limit=10)
                
                if not feedback:
                    console.print("[yellow]No feedback recorded yet.[/yellow]")
                    continue
                
                console.print("\n[bold blue]💬 Recent Feedback:[/bold blue]")
                
                for fb in feedback:
                    rating_str = "★" * fb['rating'] + "☆" * (5 - fb['rating']) if fb['rating'] else "No rating"
                    
                    feedback_panel = Panel(
                        f"[bold]Query:[/bold] {fb['query']}\n\n"
                        f"[bold]Rating:[/bold] {rating_str}\n"
                        f"{f'[bold]Feedback:[/bold] {fb['feedback']}' if fb['feedback'] else ''}\n\n"
                        f"[dim]Date: {fb['created_at']}[/dim]",
                        border_style="blue",
                        title=f"Feedback #{fb['id']}"
                    )
                    console.print(feedback_panel)
                continue
                
            if user_input.lower().startswith('search '):
                search_term = user_input[7:].strip()
                if not search_term:
                    console.print("[yellow]Please provide a search term.[/yellow]")
                    continue
                    
                console.print(f"\n[bold cyan]🔍 Searching for:[/bold cyan] {search_term}")
                
                with Progress() as progress:
                    search_task = progress.add_task("[green]Searching...", total=100)
                    progress.update(search_task, completed=50)
                    results = retriever.get_relevant_documents(search_term, top_k=5)
                    progress.update(search_task, completed=100)
                
                if not results:
                    console.print("[yellow]No relevant documents found.[/yellow]")
                    continue
                    
                console.print("\n[bold blue]Top relevant document chunks:[/bold blue]")
                
                for i, result in enumerate(results):
                    result_panel = Panel(
                        f"[bold]Document:[/bold] {result['document_title']} (Type: {result['document_type']})\n"
                        f"[bold]Relevance:[/bold] {result['similarity']:.2f}\n\n"
                        f"[bold]Content:[/bold]\n{result['chunk_text'][:300]}...",
                        border_style="blue",
                        title=f"Result #{i+1}"
                    )
                    console.print(result_panel)
                continue
            
            if user_input.lower() == 'strategy':
                console.print("\n[bold blue]Strategy Recommendation Generator[/bold blue]")
                console.print("[dim]Please provide the following inputs to generate a customized strategy recommendation:[/dim]\n")
                
                # Collect strategic inputs with validation
                risk_tolerance = Prompt.ask(
                    "Risk Tolerance", 
                    choices=["High", "Medium", "Low"], 
                    default="Medium"
                )
                
                strategic_priorities = Prompt.ask("Strategic Priorities")
                strategic_constraints = Prompt.ask("Strategic Constraints")
                execution_priorities = Prompt.ask("Execution Priorities")
                execution_constraints = Prompt.ask("Execution Constraints")
                
                strategic_inputs = {
                    'risk_tolerance': risk_tolerance,
                    'strategic_priorities': strategic_priorities,
                    'strategic_constraints': strategic_constraints,
                    'execution_priorities': execution_priorities,
                    'execution_constraints': execution_constraints
                }
                
                console.print("\n[bold cyan]⏳ Generating strategy recommendation...[/bold cyan]")
                
                with Progress() as progress:
                    gen_task = progress.add_task("[green]Analyzing business data...", total=100)
                    
                    # Update progress periodically to show activity
                    for i in range(1, 5):
                        time.sleep(0.5)
                        progress.update(gen_task, completed=i * 20)
                        progress.update(gen_task, description=f"[green]Phase {i}: {['Analyzing data', 'Evaluating context', 'Formulating strategy', 'Finalizing recommendations'][i-1]}...")
                    
                    start_time = time.time()
                    recommendation = rag_system.generate_strategy_recommendation(strategic_inputs)
                    elapsed_time = time.time() - start_time
                    
                    # Complete the progress
                    progress.update(gen_task, completed=100)
                
                console.print(f"\n[bold green]📊 STRATEGY RECOMMENDATION[/bold green] [dim](generated in {elapsed_time:.2f} seconds)[/dim]")
                
                recommendation_panel = Panel(
                    recommendation,
                    border_style="green",
                    title="QmiracTM Strategy Recommendation",
                    subtitle=f"Risk Tolerance: {risk_tolerance}"
                )
                console.print(recommendation_panel)
                
                # Save strategy to file
                save_response = Confirm.ask("\nSave this strategy recommendation to file?")
                if save_response:
                    timestamp = time.time()
                    formatted_timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"strategy_recommendation_{formatted_timestamp}.txt"
                    
                    with open(filename, 'w') as f:
                        f.write(recommendation)
                    
                    console.print(f"[green]Strategy saved to {filename}[/green]")
                    
                    # Offer PDF generation
                    pdf_option = Confirm.ask("Would you like to generate a PDF version?")
                    if pdf_option:
                        try:
                            pdf_path = strategy_output_generator.generate_pdf(
                                recommendation, 
                                strategic_inputs,
                                filename=f"strategy_recommendation_{formatted_timestamp}.pdf"
                            )
                            console.print(f"[green]Strategy PDF generated at: {pdf_path}[/green]")
                        except Exception as e:
                            console.print(f"[red]Error generating PDF: {e}[/red]")
                
                # Get feedback
                feedback = Confirm.ask("\nWould you like to provide feedback on this strategy?")
                if feedback:
                    rating = Prompt.ask("Rating (1-5 stars)", choices=["1", "2", "3", "4", "5"])
                    feedback_text = Prompt.ask("Comments (optional)")
                    
                    try:
                        rating = int(rating)
                        db.store_feedback("Strategy generation", recommendation, feedback_text, rating)
                        console.print("[green]Thank you for your feedback![/green]")
                    except ValueError:
                        console.print("[yellow]Invalid rating. Feedback not stored.[/yellow]")
            else:
                # Process regular query
                console.print("\n[bold cyan]⏳ Processing query...[/bold cyan]")
                
                with Progress() as progress:
                    query_task = progress.add_task("[green]Thinking...", total=100)
                    
                    # Update progress periodically to show activity
                    for i in range(1, 5):
                        time.sleep(0.3)
                        progress.update(query_task, completed=i * 20)
                    
                    start_time = time.time()
                    response = rag_system.process_query(user_input)
                    elapsed_time = time.time() - start_time
                    
                    # Complete the progress
                    progress.update(query_task, completed=100)
                
                response_panel = Panel(
                    response,
                    border_style="blue",
                    title="🤖 QmiracTM AI Response",
                    subtitle=f"Generated in {elapsed_time:.2f} seconds"
                )
                console.print(response_panel)
                
                # Get feedback (optional)
                feedback_response = Confirm.ask("\nWas this response helpful?")
                if not feedback_response:
                    feedback = Prompt.ask("What could be improved? (optional)")
                    rating = 2  # Below average rating
                    if feedback:
                        db.store_feedback(user_input, response, feedback, rating)
                        console.print("[green]Thank you for your feedback![/green]")
                else:
                    rating_str = Prompt.ask("How would you rate this response (1-5)?", choices=["1", "2", "3", "4", "5"])
                    try:
                        rating = int(rating_str)
                        db.store_feedback(user_input, response, "Helpful", rating)
                        console.print("[green]Thank you for your feedback![/green]")
                    except ValueError:
                        console.print("[yellow]Invalid rating. Feedback not stored.[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error processing query:[/bold red] {e}")
            import traceback
            console.print(traceback.format_exc())
    
    console.print("\n[bold blue]QmiracTM AI Assistant shutting down.[/bold blue]")

def run_demo_mode(rag_system, strategy_output_generator):
    """Run a demonstration with sample queries."""
    demo_queries = [
        "What factors should I consider in market attractiveness assessment?",
        "How can I improve my business's strategic position?",
        "What are the key components of a SWOT analysis?",
        "How should I formulate my execution priorities?",
        "What's the relationship between risk tolerance and strategic planning?"
    ]
    
    width = shutil.get_terminal_size().columns
    
    console.print("\n" + "=" * width)
    console.print("[bold blue]QmiracTM AI Assistant Demo Mode[/bold blue]", justify="center")
    console.print("=" * width)
    console.print("[dim]Running demonstration with sample business strategy queries.[/dim]")
    console.print("[yellow]Press Ctrl+C at any time to exit the demo.[/yellow]")
    
    try:
        for i, query in enumerate(demo_queries):
            console.print(f"\n\n[bold]Demo Query {i+1}/{len(demo_queries)}:[/bold]")
            console.print(f"[bold cyan]👤 {query}[/bold cyan]")
            console.print("\n[bold cyan]⏳ Processing...[/bold cyan]")
            
            with Progress() as progress:
                query_task = progress.add_task("[green]Thinking...", total=100)
                
                # Update progress periodically to show activity
                for j in range(1, 5):
                    time.sleep(0.3)
                    progress.update(query_task, completed=j * 20)
                
                start_time = time.time()
                response = rag_system.process_query(query)
                elapsed_time = time.time() - start_time
                
                # Complete the progress
                progress.update(query_task, completed=100)
            
            response_panel = Panel(
                response,
                border_style="blue", 
                title="🤖 QmiracTM AI Response",
                subtitle=f"Generated in {elapsed_time:.2f} seconds"
            )
            console.print(response_panel)
            
            # Pause between queries
            if i < len(demo_queries) - 1:
                Prompt.ask("\n[dim]Press Enter to continue to the next query...[/dim]")
    
        # Demo strategy generation
        console.print("\n\n" + "=" * width)
        console.print("[bold blue]Demo Strategy Generation[/bold blue]", justify="center")
        console.print("=" * width)
        
        console.print("\n[dim]Generating a sample business strategy with default inputs...[/dim]")
        strategic_inputs = {
            'risk_tolerance': 'Medium',
            'strategic_priorities': 'Market expansion, product innovation, customer retention',
            'strategic_constraints': 'Limited capital, competitive market',
            'execution_priorities': 'Sales growth, operational efficiency',
            'execution_constraints': 'Resource limitations, regulatory requirements'
        }
        
        # Display the inputs in a nice table
        inputs_table = Table(title="Strategic Inputs", show_header=True, header_style="bold magenta")
        inputs_table.add_column("Parameter", style="cyan")
        inputs_table.add_column("Value", style="green")
        
        for key, value in strategic_inputs.items():
            inputs_table.add_row(key.replace('_', ' ').title(), value)
        
        console.print(inputs_table)
        
        console.print("\n[bold cyan]⏳ Generating strategy recommendation...[/bold cyan]")
        
        with Progress() as progress:
            gen_task = progress.add_task("[green]Analyzing business data...", total=100)
            
            # Update progress periodically to show activity
            for i in range(1, 5):
                time.sleep(0.5)
                progress.update(gen_task, completed=i * 20)
                progress.update(gen_task, description=f"[green]Phase {i}: {['Analyzing data', 'Evaluating context', 'Formulating strategy', 'Finalizing recommendations'][i-1]}...")
            
            start_time = time.time()
            recommendation = rag_system.generate_strategy_recommendation(strategic_inputs)
            elapsed_time = time.time() - start_time
            
            # Complete the progress
            progress.update(gen_task, completed=100)
        
        recommendation_panel = Panel(
            recommendation,
            border_style="green",
            title="📊 QmiracTM Strategy Recommendation",
            subtitle=f"Generated in {elapsed_time:.2f} seconds"
        )
        console.print(recommendation_panel)
        
        # Offer to generate PDF
        pdf_option = Confirm.ask("\nWould you like to generate a PDF version?")
        if pdf_option:
            try:
                formatted_timestamp = time.strftime("%Y%m%d_%H%M%S")
                pdf_path = strategy_output_generator.generate_pdf(
                    recommendation, 
                    strategic_inputs,
                    filename=f"demo_strategy_{formatted_timestamp}.pdf"
                )
                console.print(f"[green]Strategy PDF generated at: {pdf_path}[/green]")
            except Exception as e:
                console.print(f"[red]Error generating PDF: {e}[/red]")
        
        console.print("\n[bold green]Demo completed![/bold green] Switching to interactive mode...\n")
        if Confirm.ask("Would you like to continue to interactive mode?"):
            run_interactive_mode(rag_system, None, None, strategy_output_generator)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error in demo mode:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    main()