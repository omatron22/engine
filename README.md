# QmiracTM AI-Driven Knowledge Base

**Business Strategy Intelligence System**

## Overview

QmiracTM AI-Driven Knowledge Base is an intelligent system that dynamically gathers, organizes, and synthesizes business data to generate strategic scenarios and recommendations. This system utilizes a state-of-the-art Retrieval-Augmented Generation (RAG) approach, running entirely offline to ensure data security and privacy.

The system analyzes your business's strategic assessment data and generates tailored recommendations based on specified risk tolerance, priorities, and constraints.

## Features

- **Self-Hosted & Offline**: All processing happens locally - no data leaves your system
- **Intelligent Document Processing**: Ingests PDF, CSV, and other business strategy documents
- **Semantic Search**: Find relevant information quickly across your business data
- **Strategic Analysis**: Generate comprehensive business strategy recommendations
- **Customizable Outputs**: Export recommendations as formatted PDF documents
- **Interactive Interface**: User-friendly command-line and web interfaces

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

2. **Demo Mode**:

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

## Integrating with FlutterFlow

This system is designed to be integrated with FlutterFlow applications. The core RAG functionality can be exposed through:

1. **REST API**: The web interface can be extended to provide API endpoints for FlutterFlow
2. **Direct Database Integration**: The SQLite database can be shared with FlutterFlow
3. **Code Export**: Core functionality can be exported as modules for integration

## Troubleshooting

- **Ollama Connection Issues**: Ensure Ollama is running with `ollama serve`
- **Model Not Found**: Run `ollama pull deepseek-coder:reasoning`
- **Database Errors**: Try running with `--optimize` flag
- **Memory Issues**: Close other applications or increase system swap space

## License

Proprietary - QmiracTM Business Intelligence Systems