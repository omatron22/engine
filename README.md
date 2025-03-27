# QmiracTM AI-Driven Knowledge Base

**Business Strategy Intelligence System**

## Overview

QmiracTM AI-Driven Knowledge Base is an intelligent system that dynamically gathers, organizes, and synthesizes business data to generate strategic scenarios and recommendations. This system utilizes a state-of-the-art Retrieval-Augmented Generation (RAG) approach, running entirely offline to ensure data security and privacy.

The system analyzes your business's strategic assessment data and generates tailored recommendations based on specified risk tolerance, priorities, and constraints.

## Features

- **Self-Hosted & Offline**: All processing happens locally - no data leaves your system
- **Intelligent Document Processing**: Ingests PDF, CSV, JSON, and TXT files
- **Semantic Search**: Find relevant information quickly across your business data
- **Strategic Analysis**: Generate comprehensive business strategy recommendations
- **Customizable Outputs**: Export recommendations as formatted PDF documents
- **Interactive Interface**: User-friendly command-line interface with rich formatting

## Architecture

The QmiracTM Knowledge Base system is built on a modular architecture:

Document Loader → Database  ←  User Interface
      ↓               ↓             ↓
  Embedding   →   Retriever  ←  RAG System
  Generator          ↓             ↓
                LLM Manager  →  PDF Generator

                
## Requirements

- Python 3.8 or higher
- Ollama (for local LLM hosting)
- At least 8GB RAM
- 10GB free disk space for models and database

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-org/qmirac-ai-kb.git
cd qmirac-ai-kb
```

2. **Set up a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install Ollama**

Download Ollama from [https://ollama.ai](https://ollama.ai) and follow installation instructions for your operating system.

5. **Start Ollama server**

```bash
ollama serve
```

6. **Pull required models**

```bash
ollama pull deepseek-coder:reasoning
ollama pull llama3
```

## Usage

### Running the Application

```bash
python app.py
```

### Command Line Options

- `--db-path PATH`: Specify a custom database path
- `--reload`: Force reload documents even if already in database
- `--backup`: Create a database backup before starting
- `--optimize`: Optimize database before starting
- `--demo`: Run in demo mode with sample queries

### Preparing Your Data

Place your business documents in the appropriate directories:

- `data/pdf/`: PDF documents (strategy assessments, reports)
- `data/csv/`: CSV data tables (metric data)
- `data/json/`: JSON data files
- `data/txt/`: Text documents

### Available Commands

Once the application is running, you can use the following commands:

- `help`: Display help information
- `docs`: List all loaded documents
- `search [term]`: Search for specific information
- `strategy`: Generate a comprehensive strategy recommendation
- `feedback`: View recent user feedback
- `exit`: Quit the application

## Generating Strategy Recommendations

The system will prompt you for the following inputs:

1. **Risk Tolerance**: High, Medium, or Low
2. **Strategic Priorities**: Key priorities for the business
3. **Strategic Constraints**: Limitations to consider
4. **Execution Priorities**: Operational focus areas
5. **Execution Constraints**: Operational limitations

Based on these inputs and your loaded business data, the system will generate a comprehensive strategy recommendation.

## Demo Mode

To see the system in action with sample queries and data, run:

```bash
python app.py --demo
```

This will walk you through example queries and strategy generation scenarios.

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Model Not Found**: Run `ollama pull deepseek-coder:reasoning`
- **Database Errors**: Try running with `--optimize` flag
- **Memory Issues**: Close other applications or increase system swap space

## License

Proprietary - QmiracTM Business Intelligence Systems

Copyright © 2025 Your Company