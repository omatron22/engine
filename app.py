#!/usr/bin/env python3
"""
QmiracTM AI-Driven Knowledge Base

A polished Retrieval-Augmented Generation (RAG) system for offline business strategy development.
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

# Import from package
from src.db import Database
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.retriever import Retriever
from src.llm import LLMManager
from src.rag import RAGSystem
from src.output_generator import StrategyOutputGenerator
from src.config import (
    PDF_DIR, CSV_DIR, DB_PATH, 
    RISK_TOLERANCE_LEVELS
)

# Add at the top after imports
print("Starting application...")

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
        "--optimize", 
        action="store_true",
        help="Optimize database before starting"
    )
    parser.add_argument(
        "--demo", 
        action="store_true",
        help="Run in demo mode with sample queries"
    )
    parser.add_argument(
        "--web", 
        action="store_true",
        help="Start the web interface"
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

def create_knowledge_base(db_path=DB_PATH):
    """Create and return a complete knowledge base system."""
    # Initialize components
    db = Database(db_path)
    embedding_generator = EmbeddingGenerator()
    llm_manager = LLMManager()
    document_loader = DocumentLoader(db)
    retriever = Retriever(db, embedding_generator)
    rag_system = RAGSystem(db, retriever, llm_manager)
    
    return db, document_loader, embedding_generator, retriever, llm_manager, rag_system

def load_documents(document_loader, embedding_generator, db):
    """Load documents from all supported directories."""
    start_time = time.time()
    total_docs = 0
    
    console.print("\n[bold blue]Loading Documents[/bold blue]")
    
    # Process PDF documents first (strategy assessments)
    if any(Path(PDF_DIR).glob('*')):
        console.print(f"\n[cyan]Processing PDF documents from {PDF_DIR}...[/cyan]")
        
        with Progress() as progress:
            task = progress.add_task(f"[green]Loading PDF documents...", total=100)
            progress.update(task, completed=50)
            stats = document_loader.load_and_process(PDF_DIR, embedding_generator)
            progress.update(task, completed=100)
        
        if stats["success"]:
            total_docs += stats["documents_loaded"]
            console.print(f"✅ Successfully processed [bold green]{stats['documents_loaded']}[/bold green] PDF documents")
    
    # Process CSV documents (metric data)
    if any(Path(CSV_DIR).glob('*')):
        console.print(f"\n[cyan]Processing CSV documents from {CSV_DIR}...[/cyan]")
        
        with Progress() as progress:
            task = progress.add_task(f"[green]Loading CSV documents...", total=100)
            progress.update(task, completed=50)
            stats = document_loader.load_and_process(CSV_DIR, embedding_generator)
            progress.update(task, completed=100)
        
        if stats["success"]:
            total_docs += stats["documents_loaded"]
            console.print(f"✅ Successfully processed [bold green]{stats['documents_loaded']}[/bold green] CSV documents")
    
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
                    choices=RISK_TOLERANCE_LEVELS, 
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
                feedback = Confirm.ask("\nWas this strategy recommendation helpful?")
                if feedback:
                    db.store_feedback(
                        "Strategy generation", 
                        recommendation, 
                        "User found it helpful", 
                        5
                    )
                    console.print("[green]Thank you for your feedback![/green]")
                else:
                    feedback_text = Prompt.ask("What could be improved?")
                    db.store_feedback(
                        "Strategy generation", 
                        recommendation, 
                        feedback_text, 
                        3
                    )
                    console.print("[green]Thank you for your feedback! We'll use it to improve future recommendations.[/green]")
                
                continue
            
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
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error processing query:[/bold red] {e}")
    
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
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error in demo mode:[/bold red] {e}")

def run_web_interface(db, rag_system, strategy_output_generator):
    """Run a simple web interface for the knowledge base."""
    try:
        import flask
        from flask import Flask, request, jsonify, render_template, send_from_directory
        from werkzeug.utils import secure_filename
        import threading
        import webbrowser
    except ImportError:
        console.print("[bold red]Error:[/bold red] Flask is required for the web interface.")
        console.print("Install it with: pip install flask")
        return
    
    # Create uploads folder if it doesn't exist
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    PDF_OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "strategy_outputs")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(PDF_OUTPUT_FOLDER, exist_ok=True)
    
    # Create a simple Flask app
    app = Flask("QmiracTM_KB", 
                template_folder=os.path.join(os.path.dirname(__file__), "templates"),
                static_folder=os.path.join(os.path.dirname(__file__), "static"))
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['PDF_OUTPUT_FOLDER'] = PDF_OUTPUT_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
    
    # Ensure directories exist
    os.makedirs(os.path.join(app.static_folder), exist_ok=True)
    os.makedirs(os.path.join(app.template_folder), exist_ok=True)
    
    # Initialize document_loader and embedding_generator for uploads
    document_loader = DocumentLoader(db)
    embedding_generator = EmbeddingGenerator()
    
    # Set up Flask routes
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/query', methods=['POST'])
    def query():
        data = request.json
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({'error': 'No query provided'}), 400
        
        try:
            response = rag_system.process_query(user_query)
            return jsonify({'response': response})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/strategy', methods=['POST'])
    def strategy():
        data = request.json
        
        # Validate inputs
        if not data.get('risk_tolerance'):
            return jsonify({'error': 'Risk tolerance is required'}), 400
        
        try:
            recommendation = rag_system.generate_strategy_recommendation(data)
            
            # Generate PDF
            formatted_timestamp = time.strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"strategy_recommendation_{formatted_timestamp}.pdf"
            pdf_path = strategy_output_generator.generate_pdf(
                recommendation, 
                data,
                filename=pdf_filename
            )
            
            return jsonify({
                'recommendation': recommendation,
                'pdf_path': pdf_path,
                'pdf_filename': pdf_filename
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/documents', methods=['GET'])
    def get_documents():
        try:
            documents = db.get_documents(limit=100)
            return jsonify({'documents': documents})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the uploaded file based on its type
            file_ext = os.path.splitext(filename)[1].lower()
            
            try:
                if file_ext == '.pdf':
                    doc = document_loader.load_pdf(file_path)
                elif file_ext == '.csv':
                    doc = document_loader.load_csv(file_path)
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'Unsupported file type. Please upload PDF or CSV.'
                    }), 400
                
                if doc:
                    # Generate embeddings
                    document_id = doc['id']
                    document_type = doc['type']
                    document_content = doc['content']
                    
                    embeddings = embedding_generator.generate_embeddings(
                        document_id, 
                        document_content, 
                        document_type
                    )
                    
                    # Store embeddings
                    for emb in embeddings:
                        db.store_embedding(
                            emb['document_id'],
                            emb['chunk_text'],
                            emb['embedding_vector'],
                            emb['chunk_index']
                        )
                    
                    return jsonify({
                        'success': True, 
                        'message': f'File uploaded and processed successfully. Generated {len(embeddings)} embeddings.',
                        'document_id': document_id
                    })
                else:
                    return jsonify({
                        'success': False, 
                        'message': 'Failed to process the uploaded file.'
                    }), 500
            
            except Exception as e:
                return jsonify({
                    'success': False, 
                    'message': f'Error processing file: {str(e)}'
                }), 500
    
    @app.route('/download/<filename>', methods=['GET'])
    def download_pdf(filename):
        return send_from_directory(app.config['PDF_OUTPUT_FOLDER'], filename, as_attachment=True)
    
    # Create a simple index.html if it doesn't exist
    index_path = os.path.join(app.template_folder, "index.html")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>QmiracTM Knowledge Base</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        body { padding-top: 20px; }
        .response-area { min-height: 200px; }
        .loading { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <div class="row mb-4">
            <div class="col-12 text-center">
                <h1>QmiracTM AI-Driven Knowledge Base</h1>
                <p class="lead">Business Strategy Intelligence System</p>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <ul class="nav nav-tabs card-header-tabs" id="nav-tab" role="tablist">
                            <li class="nav-item">
                                <a class="nav-link active" id="nav-query-tab" data-bs-toggle="tab" href="#nav-query" role="tab">Query</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="nav-strategy-tab" data-bs-toggle="tab" href="#nav-strategy" role="tab">Strategy Generator</a>
                            </li>
                            <li class="nav-item">
                                <a class="nav-link" id="nav-documents-tab" data-bs-toggle="tab" href="#nav-documents" role="tab">Documents</a>
                            </li>
                        </ul>
                    </div>
                    <div class="card-body">
                        <div class="tab-content" id="nav-tabContent">
                            <!-- Query Tab -->
                            <div class="tab-pane fade show active" id="nav-query" role="tabpanel">
                                <form id="query-form">
                                    <div class="mb-3">
                                        <label for="query-input" class="form-label">Ask a business strategy question:</label>
                                        <input type="text" class="form-control" id="query-input" placeholder="How can I improve my strategic position?">
                                    </div>
                                    <button type="submit" class="btn btn-primary">Submit Query</button>
                                    <div class="spinner-border text-primary loading" id="query-loading" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </form>
                                <div class="mt-4">
                                    <h5>Response:</h5>
                                    <div class="border p-3 response-area bg-light" id="query-response">
                                        <p class="text-muted">Your response will appear here.</p>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Strategy Generator Tab -->
                            <div class="tab-pane fade" id="nav-strategy" role="tabpanel">
                                <form id="strategy-form">
                                    <div class="mb-3">
                                        <label for="risk-tolerance" class="form-label">Risk Tolerance:</label>
                                        <select class="form-select" id="risk-tolerance">
                                            <option value="Low">Low</option>
                                            <option value="Medium" selected>Medium</option>
                                            <option value="High">High</option>
                                        </select>
                                    </div>
                                    <div class="mb-3">
                                        <label for="strategic-priorities" class="form-label">Strategic Priorities:</label>
                                        <textarea class="form-control" id="strategic-priorities" rows="2" placeholder="Market expansion, product innovation, customer retention"></textarea>
                                    </div>
                                    <div class="mb-3">
                                        <label for="strategic-constraints" class="form-label">Strategic Constraints:</label>
                                        <textarea class="form-control" id="strategic-constraints" rows="2" placeholder="Limited capital, competitive market"></textarea>
                                    </div>
                                    <div class="mb-3">
                                        <label for="execution-priorities" class="form-label">Execution Priorities:</label>
                                        <textarea class="form-control" id="execution-priorities" rows="2" placeholder="Sales growth, operational efficiency"></textarea>
                                    </div>
                                    <div class="mb-3">
                                        <label for="execution-constraints" class="form-label">Execution Constraints:</label>
                                        <textarea class="form-control" id="execution-constraints" rows="2" placeholder="Resource limitations, regulatory requirements"></textarea>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Generate Strategy</button>
                                    <div class="spinner-border text-primary loading" id="strategy-loading" role="status">
                                        <span class="visually-hidden">Loading...</span>
                                    </div>
                                </form>
                                <div class="mt-4">
                                    <h5>Strategy Recommendation:</h5>
                                    <div class="border p-3 response-area bg-light" id="strategy-response">
                                        <p class="text-muted">Your strategy recommendation will appear here.</p>
                                    </div>
                                    <button class="btn btn-success mt-2" id="download-pdf" style="display: none;">Download PDF</button>
                                </div>
                            </div>
                            
                            <!-- Documents Tab -->
                            <div class="tab-pane fade" id="nav-documents" role="tabpanel">
                                <div class="mb-3">
                                    <h5>Upload Documents</h5>
                                    <form id="upload-form" enctype="multipart/form-data">
                                        <div class="mb-3">
                                            <label for="document-file" class="form-label">Select PDF or CSV file:</label>
                                            <input class="form-control" type="file" id="document-file" name="file" accept=".pdf,.csv">
                                        </div>
                                        <button type="submit" class="btn btn-primary">Upload</button>
                                        <div class="spinner-border text-primary loading" id="upload-loading" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                    </form>
                                </div>
                                <div class="mt-4">
                                    <h5>Loaded Documents</h5>
                                    <button id="refresh-docs" class="btn btn-outline-secondary btn-sm mb-2">Refresh List</button>
                                    <div id="documents-list" class="border p-3 bg-light">
                                        <p class="text-muted">Loading documents...</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Query Tab
        document.getElementById('query-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const query = document.getElementById('query-input').value;
            if (!query) return;
            
            document.getElementById('query-loading').style.display = 'inline-block';
            document.getElementById('query-response').innerHTML = '<p>Processing query...</p>';
            
            fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('query-loading').style.display = 'none';
                document.getElementById('query-response').innerHTML = `<p>${data.response.replace(/\\n/g, '<br>')}</p>`;
            })
            .catch(error => {
                document.getElementById('query-loading').style.display = 'none';
                document.getElementById('query-response').innerHTML = `<p class="text-danger">Error: ${error}</p>`;
            });
        });
        
        // Strategy Generator Tab
        document.getElementById('strategy-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const strategic_inputs = {
                risk_tolerance: document.getElementById('risk-tolerance').value,
                strategic_priorities: document.getElementById('strategic-priorities').value,
                strategic_constraints: document.getElementById('strategic-constraints').value,
                execution_priorities: document.getElementById('execution-priorities').value,
                execution_constraints: document.getElementById('execution-constraints').value
            };
            
            document.getElementById('strategy-loading').style.display = 'inline-block';
            document.getElementById('strategy-response').innerHTML = '<p>Generating strategy recommendation...</p>';
            document.getElementById('download-pdf').style.display = 'none';
            
            fetch('/api/strategy', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(strategic_inputs)
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('strategy-loading').style.display = 'none';
                document.getElementById('strategy-response').innerHTML = `<p>${data.recommendation.replace(/\\n/g, '<br>')}</p>`;
                
                if (data.pdf_path) {
                    document.getElementById('download-pdf').style.display = 'inline-block';
                    document.getElementById('download-pdf').onclick = function() {
                        window.location.href = `/download/${data.pdf_filename}`;
                    };
                }
            })
            .catch(error => {
                document.getElementById('strategy-loading').style.display = 'none';
                document.getElementById('strategy-response').innerHTML = `<p class="text-danger">Error: ${error}</p>`;
            });
        });
        
        // Documents Tab
        function loadDocuments() {
            fetch('/api/documents')
            .then(response => response.json())
            .then(data => {
                const docsList = document.getElementById('documents-list');
                if (data.documents.length === 0) {
                    docsList.innerHTML = '<p class="text-muted">No documents loaded.</p>';
                    return;
                }
                
                let html = '<div class="list-group">';
                data.documents.forEach(doc => {
                    html += `<div class="list-group-item list-group-item-action">
                        <div class="d-flex w-100 justify-content-between">
                            <h6 class="mb-1">${doc.title}</h6>
                            <small>ID: ${doc.id}</small>
                        </div>
                        <p class="mb-1">Type: ${doc.document_type}</p>
                    </div>`;
                });
                html += '</div>';
                docsList.innerHTML = html;
            })
            .catch(error => {
                document.getElementById('documents-list').innerHTML = `<p class="text-danger">Error loading documents: ${error}</p>`;
            });
        }
        
        document.getElementById('refresh-docs').addEventListener('click', loadDocuments);
        
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('document-file');
            const file = fileInput.files[0];
            if (!file) {
                alert('Please select a file to upload');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            document.getElementById('upload-loading').style.display = 'inline-block';
            
            fetch('/api/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('upload-loading').style.display = 'none';
                if (data.success) {
                    alert('Document uploaded successfully!');
                    fileInput.value = '';
                    loadDocuments();
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(error => {
                document.getElementById('upload-loading').style.display = 'none';
                alert('Error uploading document: ' + error);
            });
        });
        
        // Load documents when tab is shown
        document.getElementById('nav-documents-tab').addEventListener('click', loadDocuments);
        
        // Initial document load
        window.onload = loadDocuments;
    </script>
</body>
</html>
""")
            

def main():
    """Main execution function."""
    print("Arguments parsed...")
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    args = setup_argparse()
    
    # Display welcome banner
    display_welcome_banner()
    
    print("Creating knowledge base...")
    # Create knowledge base components
    db, document_loader, embedding_generator, retriever, llm_manager, rag_system = create_knowledge_base(args.db_path)
    
    # Optimize database if requested
    if args.optimize:
        console.print("\n[cyan]Optimizing database...[/cyan]")
        db.vacuum()
    
    # Initialize output generator
    strategy_output_generator = StrategyOutputGenerator()
    
    print("Loading documents...")
    # Load documents if needed (either --reload flag or empty database)
    if args.reload or db.get_embeddings_count() == 0:
        load_documents(document_loader, embedding_generator, db)
    else:
        console.print(f"\n[cyan]Using existing document embeddings. ({db.get_embeddings_count()} embeddings found)[/cyan]")
        console.print("[dim]Use --reload to force document reloading[/dim]")
    
    print("Running application mode...")
    # Run the appropriate mode
    if args.web:
        console.print("\n[bold blue]Starting Web Interface[/bold blue]")
        run_web_interface(db, rag_system, strategy_output_generator)
    elif args.demo:
        console.print("\n[bold blue]Starting Demo Mode[/bold blue]")
        run_demo_mode(rag_system, strategy_output_generator)
        # After demo, switch to interactive mode
        run_interactive_mode(rag_system, db, retriever, strategy_output_generator)
    else:
        run_interactive_mode(rag_system, db, retriever, strategy_output_generator)
    
    # Close the database connection
    db.close()

if __name__ == "__main__":
    try:
        print("Entering main execution...")
        main()
    except Exception as e:
        print(f"ERROR: Application failed with exception: {e}")
        import traceback
        traceback.print_exc()