# QmiracTM AI-Driven Knowledge Base

**Business Strategy Intelligence System**

## Overview

QmiracTM AI-Driven Knowledge Base is an intelligent system that dynamically gathers, organizes, and synthesizes business data to generate strategic scenarios and actionable recommendations. This system utilizes a state-of-the-art Retrieval-Augmented Generation (RAG) approach, running entirely offline to ensure data security and privacy.

The system analyzes your business's strategic assessment data and generates tailored recommendations based on your specified risk tolerance, strategic priorities, and constraints.

## Key Features

- **Self-Hosted & Offline**: All processing happens locally - no data leaves your system
- **Intelligent Document Processing**: Automatically ingests PDFs, CSVs, and other business strategy documents
- **Semantic Search**: Find relevant information quickly across your business data 
- **Strategic Analysis**: Generate comprehensive business strategy recommendations
- **Customizable Outputs**: Export recommendations as formatted PDF documents
- **Interactive Interface**: User-friendly command-line and web interfaces

## How It Works

1. **Data Ingestion**: Upload your strategic assessment documents, financial data, market analyses, and other business data
2. **Intelligent Processing**: The system processes and indexes your data using advanced embedding techniques
3. **Strategic Inputs**: Specify your risk tolerance, strategic priorities, and constraints
4. **AI-Powered Analysis**: The system analyzes your data and inputs to generate comprehensive strategy recommendations
5. **Output Generation**: Receive detailed strategy recommendations that can be exported as PDF documents

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

There are multiple ways to run the application:

1. **Command Line Interface**:

```bash
python app.py
```

2. **Demo Mode** (runs with sample data and queries):

```bash
python app.py --demo
```

3. **Web Interface**:

```bash
python app.py --web
```

### Command Line Options

- `--db-path PATH`: Specify a custom database path
- `--reload`: Force reload documents even if already in database
- `--optimize`: Optimize database before starting
- `--demo`: Run in demo mode with sample queries
- `--web`: Start the web interface

### Preparing Your Data

Place your business documents in the appropriate directories:

- `data/pdf/`: PDF documents (strategy assessments, reports)
- `data/csv/`: CSV data tables (metric data)

### Generating Strategy Recommendations

The system will prompt you for the following inputs:

1. **Risk Tolerance**: High, Medium, or Low
2. **Strategic Priorities**: Key priorities for the business
3. **Strategic Constraints**: Limitations to consider
4. **Execution Priorities**: Operational focus areas
5. **Execution Constraints**: Operational limitations

Based on these inputs and your loaded business data, the system will generate a comprehensive strategy recommendation.

## Architecture

The QmiracTM AI-Driven Knowledge Base consists of several key components:

- **Document Loader**: Processes PDFs, CSVs, and other documents
- **Embedding Generator**: Creates vector embeddings for document chunks
- **Retriever**: Finds relevant information based on queries
- **LLM Manager**: Interacts with Ollama for text generation
- **RAG System**: Coordinates the entire Retrieval-Augmented Generation process
- **Output Generator**: Creates formatted PDF strategy recommendations

## Sample Data

The repository includes sample data tables for demonstration purposes:

- Financial metrics (revenue, profit, margins)
- Market assessment scores
- Competitive analysis
- SWOT analysis components
- Operational metrics

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Model Not Found**: Run `ollama pull deepseek-coder:reasoning`
- **Database Errors**: Try running with `--optimize` flag
- **Memory Issues**: Close other applications or increase system swap space

## License

Proprietary - QmiracTM Business Intelligence Systems

---

**QmiracTM AI-Driven Knowledge Base** - Transforming Business Data into Strategic Intelligence