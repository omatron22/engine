#!/usr/bin/env python3
"""
QmiracTM AI Knowledge Base - Guided Demo Script

This script provides a guided demonstration of the QmiracTM AI Knowledge Base system
with a prepared set of examples and transitions for presentation purposes.
"""
import os
import time
import subprocess
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.markdown import Markdown
import shutil

console = Console()

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_header():
    """Display the demo header."""
    width = shutil.get_terminal_size().columns
    console.print("\n" + "=" * width)
    console.print("[bold blue]QmiracTM AI Knowledge Base[/bold blue]", justify="center")
    console.print("[bold cyan]Guided Demonstration[/bold cyan]", justify="center")
    console.print("=" * width + "\n")

def pause(message="Press Enter to continue..."):
    """Wait for user input to continue."""
    console.print(f"\n[dim]{message}[/dim]")
    input()

def show_introduction():
    """Display the introduction to the demo."""
    clear_screen()
    display_header()
    
    intro_text = """
# Welcome to QmiracTM AI Knowledge Base

This demonstration will guide you through the key capabilities of the QmiracTM AI-driven Knowledge Base system, showcasing how it can help transform your business strategy development process.

## What You'll See Today

1. **Data Ingestion**: How the system processes your business documents and data
2. **Interactive Queries**: Asking business strategy questions and getting informed answers
3. **Strategy Generation**: Creating comprehensive, tailored strategy recommendations
4. **PDF Output**: Professionally formatted strategy documentation

## Key Benefits

- **Self-Hosted & Secure**: All data processing happens locally
- **Intelligent Analysis**: Deep understanding of business strategy concepts
- **Time-Saving**: Accelerate strategy development and decision-making
- **Customizable**: Adapts to your business's specific needs and risk profile

Let's get started!
    """
    
    md = Markdown(intro_text)
    console.print(md)
    
    pause()

def simulate_typing(text, delay=0.03):
    """Simulate typing text with a delay between characters."""
    for char in text:
        console.print(char, end="")
        time.sleep(delay)
    console.print()

def run_demo_sequence():
    """Run the main demo sequence."""
    # Introduction
    show_introduction()
    
    # Part 1: System Overview
    clear_screen()
    display_header()
    console.print("[bold green]Part 1: System Overview[/bold green]\n")
    
    console.print(Panel(
        "The QmiracTM AI Knowledge Base uses a Retrieval-Augmented Generation (RAG) architecture "
        "to provide accurate, contextually relevant strategic insights. "
        "The system combines:\n\n"
        "• [bold]Vector Database[/bold]: For efficient information retrieval\n"
        "• [bold]Embedding Model[/bold]: To understand semantic meaning\n"
        "• [bold]Large Language Model[/bold]: For intelligent response generation\n"
        "• [bold]Document Processing[/bold]: To ingest various data formats",
        title="System Architecture",
        border_style="blue"
    ))
    
    pause()
    
    # Show the commands that would be run
    console.print("\n[bold cyan]System Initialization[/bold cyan]")
    console.print("\nThe following command starts the QmiracTM Knowledge Base:")
    
    cmd_syntax = Syntax("python app.py", "bash", theme="monokai", line_numbers=False)
    console.print(cmd_syntax)
    
    console.print("\nFor this demo, we'll use additional flags:")
    demo_cmd_syntax = Syntax("python app.py --demo --optimize", "bash", theme="monokai", line_numbers=False)
    console.print(demo_cmd_syntax)
    
    pause()
    
    # Part 2: Document Loading
    clear_screen()
    display_header()
    console.print("[bold green]Part 2: Document Processing[/bold green]\n")
    
    console.print("The system ingests documents from multiple sources:")
    
    table = """
┌────────────────────┬─────────────────────────────────┐
│ Document Type      │ Purpose                         │
├────────────────────┼─────────────────────────────────┤
│ Strategy PDFs      │ Strategic assessments and plans │
│ CSV Data Tables    │ Key metrics and performance data│
│ Market Reports     │ Competitive landscape analysis  │
│ SWOT Analyses      │ Strengths, weaknesses, etc.     │
│ Financial Reports  │ Revenue, costs, forecasts       │
└────────────────────┴─────────────────────────────────┘
"""
    console.print(table)
    
    console.print("\n[bold]Document Loading Process:[/bold]")
    console.print("1. Files are ingested from organized directories")
    console.print("2. Text is extracted and processed")
    console.print("3. Content is divided into semantic chunks")
    console.print("4. Embeddings are generated for each chunk")
    console.print("5. Relationships between information are established")
    
    pause()
    
    # Part 3: Interactive Query Demo
    clear_screen()
    display_header()
    console.print("[bold green]Part 3: Interactive Queries[/bold green]\n")
    
    console.print("Let's see how the system responds to business strategy questions:")
    
    sample_queries = [
        "What factors should I consider when assessing market attractiveness?",
        "How do I create an effective competitive assessment?",
        "What are the key elements of a SWOT analysis?"
    ]
    
    for i, query in enumerate(sample_queries):
        console.print(f"\n[bold cyan]Example Query {i+1}:[/bold cyan]")
        console.print(f"[bold]Q:[/bold] {query}")
        console.print("\n[bold]A:[/bold] ", end="")
        
        # Simulate an AI response
        if i == 0:
            response = """Based on the strategic assessment data, market attractiveness should be evaluated using these key factors:

1. Market Size and Growth: The total addressable market and its projected growth rate
2. Competitive Intensity: Number and strength of competitors
3. Entry Barriers: Ease for new companies to enter the market
4. Pricing Power: Ability to maintain profit margins
5. Customer Needs: Alignment with your capabilities

The BizGuru app supports quantitative assessment of these factors through the Market Assessment module, where you can assign scores and weights to each criterion."""
        elif i == 1:
            response = """To create an effective competitive assessment in QmiracTM BizGuru:

1. First identify your key business metrics (specs) that matter most to your customers
2. Input your own business's performance scores for these metrics
3. Select your primary competitors for comparison
4. Input/import competitor data for each metric
5. The system will generate comparative analysis showing your strategic position

This process helps identify your competitive advantages and disadvantages across critical performance dimensions."""
        else:
            response = """A comprehensive SWOT analysis in the QmiracTM framework includes:

• Strengths Assessment: Identify internal capabilities giving you advantage, quantify each strength
• Weaknesses Assessment: Recognize internal limitations, quantify impact of each weakness
• Opportunities Assessment: Analyze external favorable factors, quantify potential value
• Threats Assessment: Identify external challenges, quantify potential risk

The BizGuru app guides you through structured assessment of each element with quantitative evaluation, creating a data-driven foundation for strategy development."""
        
        # Simulate typing for a more dynamic demo
        simulate_typing(response, delay=0.005)
        
        if i < len(sample_queries) - 1:
            pause()
    
    pause("Press Enter to continue to Strategy Generation...")
    
    # Part 4: Strategy Generation
    clear_screen()
    display_header()
    console.print("[bold green]Part 4: Strategy Generation[/bold green]\n")
    
    console.print("Now, let's generate a complete business strategy recommendation.")
    console.print("The system will ask for several key inputs:")
    
    console.print("\n[bold]1. Risk Tolerance:[/bold] How much risk is the business willing to accept?")
    console.print("   Options: High, Medium, Low")
    
    console.print("\n[bold]2. Strategic Priorities:[/bold] What are the most important goals?")
    console.print("   Example: Market expansion, customer retention, operational efficiency")
    
    console.print("\n[bold]3. Strategic Constraints:[/bold] What limitations must be considered?")
    console.print("   Example: Limited capital, regulatory requirements, resource limitations")
    
    console.print("\n[bold]4. Execution Priorities:[/bold] Key operational focus areas")
    console.print("   Example: Sales growth, cost reduction, quality improvement")
    
    console.print("\n[bold]5. Execution Constraints:[/bold] Operational limitations")
    console.print("   Example: Team capacity, technology limitations, supply chain issues")
    
    pause()
    
    # Show sample inputs
    console.print("\n[bold cyan]Sample Strategy Generation Inputs:[/bold cyan]")
    
    inputs = {
        "Risk Tolerance": "Medium",
        "Strategic Priorities": "Market expansion in B2B segment, product innovation, improving customer retention",
        "Strategic Constraints": "Limited investment capital, intense market competition, talent acquisition challenges",
        "Execution Priorities": "Sales team efficiency, operational cost reduction, quality improvement",
        "Execution Constraints": "Legacy IT systems, production capacity limitations, regulatory compliance requirements"
    }
    
    for key, value in inputs.items():
        console.print(f"[bold]{key}:[/bold] {value}")
    
    pause()
    
    # Show the strategy generation process
    console.print("\n[bold cyan]Strategy Generation Process:[/bold cyan]")
    console.print("1. System retrieves relevant information from knowledge base")
    console.print("2. Inputs are analyzed in context of your business data")
    console.print("3. Strategic options are evaluated based on risk tolerance")
    console.print("4. Comprehensive strategy is formulated with specific recommendations")
    console.print("5. Output is generated in structured format")
    
    pause()
    
    # Part 5: PDF Output
    clear_screen()
    display_header()
    console.print("[bold green]Part 5: PDF Output Generation[/bold green]\n")
    
    console.print("The final step is generating a professional PDF document containing the strategy recommendation.")
    
    console.print(Panel(
        "The PDF report includes:\n\n"
        "• [bold]Executive Summary[/bold]: Brief overview of findings and recommendations\n"
        "• [bold]Strategic Direction[/bold]: Core positioning and value proposition\n"
        "• [bold]Key Strategic Initiatives[/bold]: Prioritized list with expected outcomes\n"
        "• [bold]Risk Assessment[/bold]: Analysis of major risks and mitigation strategies\n"
        "• [bold]Implementation Roadmap[/bold]: Phased plan with key milestones\n"
        "• [bold]Success Factors & KPIs[/bold]: Measurable outcomes and evaluation framework",
        title="Strategy Recommendation Document",
        border_style="green"
    ))
    
    console.print("\nThis document provides a comprehensive strategic plan that can be shared with stakeholders, implementation teams, and executives.")
    
    # Show the command to generate a PDF
    console.print("\n[bold]To save the strategy as a PDF:[/bold]")
    console.print("The system will ask if you want to save the strategy recommendation to a PDF file.")
    console.print("Selecting 'yes' will create a professionally formatted document.")
    
    pause()
    
    # Conclusion
    clear_screen()
    display_header()
    console.print("[bold green]Demo Conclusion[/bold green]\n")
    
    conclusion_text = """
# QmiracTM AI Knowledge Base - Key Benefits

## Strategic Value
- **Data-Driven Decisions**: Base strategy on quantitative assessment, not just intuition
- **Comprehensive Analysis**: Consider all relevant factors in your strategic planning
- **Risk-Appropriate Planning**: Tailor recommendations to your risk tolerance level
- **Execution Focus**: Bridge the gap between strategy and implementation

## Operational Efficiency
- **Time Savings**: Reduce strategy development time by 60-70%
- **Resource Optimization**: Make better use of limited resources
- **Consistent Methodology**: Standardized approach to strategy development
- **Knowledge Retention**: Capture and leverage institutional knowledge

## Technical Advantages
- **Security & Privacy**: All data stays on your systems
- **Customizable**: Adapts to your business's specific context
- **Scalable**: Handles growing volumes of business data
- **Future-Proof**: Architecture designed for easy updates and enhancements

# Next Steps

- **Schedule a Technical Review**: Detailed evaluation of implementation requirements
- **Pilot Program**: Limited deployment to demonstrate value in your environment
- **Full Implementation Plan**: Comprehensive roadmap for organization-wide adoption
- **Training & Support**: Ensure your team maximizes the system's capabilities
    """
    
    md = Markdown(conclusion_text)
    console.print(md)
    
    console.print("\n[bold blue]Thank you for attending this demonstration of QmiracTM AI Knowledge Base![/bold blue]")
    console.print("[cyan]Questions and discussion are welcome.[/cyan]")

if __name__ == "__main__":
    try:
        run_demo_sequence()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n\n[bold red]Error in demo script:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())