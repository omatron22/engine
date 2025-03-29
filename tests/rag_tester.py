#!/usr/bin/env python3
"""
QmiracTM AI-Driven Knowledge Base Testing Framework

A framework for testing the performance and accuracy of the RAG system.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# Add the parent directory to sys.path so Python can find the src module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
print(f"Added to Python path: {parent_dir}")

from rich.console import Console
from rich.table import Table
from rich.progress import Progress

# Now we can import from src
from src.db import Database
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingGenerator
from src.retriever import Retriever
from src.llm import LLMManager
from src.rag import RAGSystem
from src.config import DB_PATH

console = Console()

class RAGTester:
    def __init__(self, db_path=DB_PATH):
        """Initialize the testing framework."""
        self.db_path = db_path
        
        # Initialize components
        self.db = Database(db_path)
        self.embedding_generator = EmbeddingGenerator()
        self.llm_manager = LLMManager()
        self.document_loader = DocumentLoader(self.db)
        self.retriever = Retriever(self.db, self.embedding_generator)
        self.rag_system = RAGSystem(self.db, self.retriever, self.llm_manager)
        
        # Test metrics
        self.metrics = {
            "retrieval": {},
            "generation": {},
            "end_to_end": {}
        }
        
    def run_tests(self, test_file="tests/test_queries.json"):
        """Run all tests defined in the test file."""
        if not os.path.exists(test_file):
            console.print(f"[bold red]Error:[/bold red] Test file {test_file} not found")
            return
            
        # Load test queries
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        console.print(f"\n[bold blue]Starting RAG System Tests[/bold blue] ({len(test_data['test_cases'])} test cases)")
        
        # Test retrieval
        self.test_retrieval(test_data["test_cases"])
        
        # Test generation
        self.test_generation(test_data["test_cases"])
        
        # Test end-to-end
        self.test_end_to_end(test_data["test_cases"])
        
        # Generate report
        self.generate_report()
        
    def test_retrieval(self, test_cases):
        """Test document retrieval component."""
        console.print("\n[bold cyan]Testing Retrieval Component[/bold cyan]")
        
        retrieval_times = []
        retrieval_counts = []
        
        with Progress() as progress:
            retrieval_task = progress.add_task("[green]Testing retrieval...", total=len(test_cases))
            
            for i, test_case in enumerate(test_cases):
                query = test_case["query"]
                expected_docs = test_case.get("expected_docs", [])
                
                # Measure retrieval time
                start_time = time.time()
                retrieved_docs = self.retriever.get_relevant_documents(query)
                elapsed_time = time.time() - start_time
                
                # Record metrics
                retrieval_times.append(elapsed_time)
                retrieval_counts.append(len(retrieved_docs))
                
                # Check if expected documents were retrieved
                if expected_docs:
                    retrieved_titles = [doc.get('document_title', '') for doc in retrieved_docs]
                    found_expected = sum(1 for doc in expected_docs if doc in retrieved_titles)
                    accuracy = found_expected / len(expected_docs) if expected_docs else 0
                    test_case["retrieval_accuracy"] = accuracy
                
                progress.update(retrieval_task, advance=1)
        
        # Calculate average metrics
        self.metrics["retrieval"] = {
            "avg_time": sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            "avg_docs": sum(retrieval_counts) / len(retrieval_counts) if retrieval_counts else 0,
            "max_time": max(retrieval_times) if retrieval_times else 0,
            "min_time": min(retrieval_times) if retrieval_times else 0
        }
        
        console.print(f"[green]Retrieval testing completed:[/green] Avg time: {self.metrics['retrieval']['avg_time']:.2f}s, Avg docs: {self.metrics['retrieval']['avg_docs']:.1f}")
    
    def test_generation(self, test_cases):
        """Test response generation component."""
        console.print("\n[bold cyan]Testing Generation Component[/bold cyan]")
        
        generation_times = []
        
        with Progress() as progress:
            generation_task = progress.add_task("[green]Testing generation...", total=len(test_cases))
            
            for i, test_case in enumerate(test_cases):
                query = test_case["query"]
                
                # First get relevant documents
                retrieved_docs = self.retriever.get_relevant_documents(query)
                
                # Prepare context for the LLM
                context_parts = []
                for doc in retrieved_docs:
                    context_parts.append(doc['chunk_text'])
                context = "\n\n".join(context_parts)
                
                # Measure generation time
                start_time = time.time()
                response = self.llm_manager.generate_response(
                    prompt=f"Based on this context:\n{context}\n\nAnswer this query: {query}",
                    model="deepseek-r1:latest"
                )
                elapsed_time = time.time() - start_time
                
                # Record metrics
                generation_times.append(elapsed_time)
                
                progress.update(generation_task, advance=1)
        
        # Calculate average metrics
        self.metrics["generation"] = {
            "avg_time": sum(generation_times) / len(generation_times) if generation_times else 0,
            "max_time": max(generation_times) if generation_times else 0,
            "min_time": min(generation_times) if generation_times else 0
        }
        
        console.print(f"[green]Generation testing completed:[/green] Avg time: {self.metrics['generation']['avg_time']:.2f}s")
    
    def test_end_to_end(self, test_cases):
        """Test the entire RAG pipeline."""
        console.print("\n[bold cyan]Testing End-to-End RAG Pipeline[/bold cyan]")
        
        total_times = []
        
        with Progress() as progress:
            e2e_task = progress.add_task("[green]Testing end-to-end...", total=len(test_cases))
            
            for i, test_case in enumerate(test_cases):
                query = test_case["query"]
                
                # Measure total processing time
                start_time = time.time()
                response = self.rag_system.process_query(query)
                elapsed_time = time.time() - start_time
                
                # Record metrics
                total_times.append(elapsed_time)
                test_case["response"] = response
                test_case["processing_time"] = elapsed_time
                
                progress.update(e2e_task, advance=1)
        
        # Calculate average metrics
        self.metrics["end_to_end"] = {
            "avg_time": sum(total_times) / len(total_times) if total_times else 0,
            "max_time": max(total_times) if total_times else 0,
            "min_time": min(total_times) if total_times else 0
        }
        
        console.print(f"[green]End-to-end testing completed:[/green] Avg time: {self.metrics['end_to_end']['avg_time']:.2f}s")
    
    def generate_report(self):
        """Generate a performance report."""
        console.print("\n[bold blue]Performance Report[/bold blue]")
        
        # Create performance table
        performance_table = Table(title="RAG System Performance Metrics")
        performance_table.add_column("Component", style="cyan")
        performance_table.add_column("Avg Time", style="green")
        performance_table.add_column("Min Time", style="blue")
        performance_table.add_column("Max Time", style="red")
        performance_table.add_column("Notes", style="yellow")
        
        # Add retrieval metrics
        performance_table.add_row(
            "Retrieval",
            f"{self.metrics['retrieval']['avg_time']:.2f}s",
            f"{self.metrics['retrieval']['min_time']:.2f}s",
            f"{self.metrics['retrieval']['max_time']:.2f}s",
            f"Avg {self.metrics['retrieval']['avg_docs']:.1f} docs retrieved"
        )
        
        # Add generation metrics
        performance_table.add_row(
            "Generation",
            f"{self.metrics['generation']['avg_time']:.2f}s",
            f"{self.metrics['generation']['min_time']:.2f}s",
            f"{self.metrics['generation']['max_time']:.2f}s",
            "Using deepseek-r1:latest"
        )
        
        # Add end-to-end metrics
        performance_table.add_row(
            "End-to-End",
            f"{self.metrics['end_to_end']['avg_time']:.2f}s",
            f"{self.metrics['end_to_end']['min_time']:.2f}s",
            f"{self.metrics['end_to_end']['max_time']:.2f}s",
            "Complete RAG pipeline"
        )
        
        console.print(performance_table)
        
        # Save metrics to file
        report_path = "test_results.json"
        with open(report_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
        console.print(f"[green]Detailed metrics saved to {report_path}[/green]")
        
        # Recommendations based on metrics
        self._provide_recommendations()
    
    def _provide_recommendations(self):
        """Provide recommendations based on test results."""
        console.print("\n[bold blue]Performance Recommendations[/bold blue]")
        
        recommendations = []
        
        # Check retrieval time
        if self.metrics["retrieval"]["avg_time"] > 1.0:
            recommendations.append(
                "Consider optimizing embeddings by reducing dimensionality or using faster similarity search"
            )
        
        # Check generation time
        if self.metrics["generation"]["avg_time"] > 3.0:
            recommendations.append(
                "Consider using a smaller/faster model for less complex queries to improve response time"
            )
        
        # Check end-to-end time
        if self.metrics["end_to_end"]["avg_time"] > 5.0:
            recommendations.append(
                "Overall pipeline is slow. Consider caching common queries or reducing context window size"
            )
        
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                console.print(f"[yellow]{i}. {rec}[/yellow]")
        else:
            console.print("[green]Performance looks good! No specific recommendations at this time.[/green]")

def create_test_file():
    """Create a sample test file if one doesn't exist."""
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_queries.json"
    
    if not test_file.exists():
        sample_tests = {
  "test_cases": [
    {
      "query": "Based on our current market assessment, what strategic direction should we take?",
      "description": "Tests ability to synthesize market data into strategic recommendations",
      "expected_docs": ["Market Assessment", "Strategic Position", "Strategic Assessment"]
    },
    {
      "query": "What are our main competitive advantages according to the data?",
      "description": "Tests ability to identify competitive strengths from assessments",
      "expected_docs": ["Competitive Assessment", "SWOT Assessment"]
    },
    {
      "query": "How does our current operational efficiency compare to industry benchmarks?",
      "description": "Tests ability to evaluate operational metrics against benchmarks",
      "expected_docs": ["Operations Dashboard", "Execution"]
    },
    {
      "query": "What financial metrics show the greatest opportunity for improvement?",
      "description": "Tests ability to identify financial optimization opportunities",
      "expected_docs": ["Finance Dashboard", "Financial Data"]
    },
    {
      "query": "Given our low risk tolerance, what market segments should we prioritize?",
      "description": "Tests consideration of risk tolerance in strategic recommendations",
      "expected_docs": ["Market Assessment", "Risk Assessment"]
    },
    {
      "query": "What are the key threats to our business identified in our strategic assessment?",
      "description": "Tests extraction of threat information from assessments",
      "expected_docs": ["SWOT Assessment", "Risk Assessment"]
    },
    {
      "query": "How should we adjust our product portfolio based on current market trends?",
      "description": "Tests product strategy recommendations based on market data",
      "expected_docs": ["Portfolio Assessment", "Market Assessment"]
    },
    {
      "query": "What is our current employee turnover rate and how does it compare to previous years?",
      "description": "Tests specific HR metric retrieval and comparison",
      "expected_docs": ["HR Dashboard", "Execution"]
    },
    {
      "query": "Create a strategic roadmap for the next 12 months based on our assessments",
      "description": "Tests complex strategy synthesis from multiple documents",
      "expected_docs": ["Strategic Assessment", "Execution", "SWOT Assessment"]
    },
    {
      "query": "What are the primary factors affecting our market attractiveness scores?",
      "description": "Tests understanding of market attractiveness methodology",
      "expected_docs": ["Market Assessment", "Market Attractiveness Data"]
    },
    {
      "query": "Generate a risk mitigation plan for our top three business risks",
      "description": "Tests risk analysis and mitigation strategy generation",
      "expected_docs": ["Risk Assessment"]
    },
    {
      "query": "How should we allocate our marketing budget based on our strategic priorities?",
      "description": "Tests budget allocation recommendations based on strategic context",
      "expected_docs": ["Sales & Marketing Dashboard", "Strategic Assessment"]
    },
    {
      "query": "What key performance indicators should we track for our strategic initiatives?",
      "description": "Tests KPI recommendation capabilities",
      "expected_docs": ["Execution", "Strategic Assessment"]
    },
    {
      "query": "Based on our SWOT analysis, what are our most promising growth opportunities?",
      "description": "Tests extraction and prioritization of growth opportunities",
      "expected_docs": ["SWOT Assessment", "Opportunities Assessment"]
    },
    {
      "query": "Develop a customer retention strategy based on our competitive position",
      "description": "Tests strategy generation based on competitive analysis",
      "expected_docs": ["Competitive Assessment", "Strategic Position"]
    }
  ]
}
        
        with open(test_file, 'w') as f:
            json.dump(sample_tests, f, indent=2)
        
        console.print(f"[green]Created sample test file at {test_file}[/green]")
    
    return test_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the RAG system performance and accuracy")
    parser.add_argument("--test-file", help="Path to test case file")
    parser.add_argument("--db-path", default=DB_PATH, help="Path to database file")
    args = parser.parse_args()
    
    # Create test file if not specified
    test_file = args.test_file if args.test_file else create_test_file()
    
    # Run tests
    tester = RAGTester(db_path=args.db_path)
    tester.run_tests(test_file=test_file)