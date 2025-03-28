# src/rag.py

import psutil
import time
from typing import Dict, List, Any, Optional

class RAGSystem:
    def __init__(self, db, retriever, llm_manager):
        """
        Initialize the RAG system for QmiracTM AI-Driven Knowledge Base.
        
        Args:
            db: Database instance for storing interactions
            retriever: Retriever instance for finding relevant documents
            llm_manager: LLM Manager for generating responses
        """
        self.db = db
        self.retriever = retriever
        self.llm = llm_manager

    def optimize_memory_usage(self, query, retrieved_docs):
        """Optimize memory usage when processing large retrievals."""
        # Limit context size based on available system memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        
        # Adjust context size based on available memory
        if available_memory < 500:  # Less than 500MB available
            # Reduce context size dramatically
            retrieved_docs = retrieved_docs[:3]
        elif available_memory < 1000:  # Less than 1GB available
            # Moderate context reduction
            retrieved_docs = retrieved_docs[:5]
        
        return retrieved_docs
    
    def process_query(self, query, additional_context=None):
        """
        Process a query through the RAG pipeline.
        
        Retrieves relevant business strategy information and generates
        a response using the language model.
        
        Args:
            query: The user's question or request
            additional_context: Optional additional context to include
            
        Returns:
            Generated response from the AI
        """
        print(f"Processing query: '{query}'")
        start_time = time.time()
        
        # Retrieve relevant documents
        relevant_docs = self.retriever.get_relevant_documents(query)
        
        # Apply memory optimization after retrieving documents
        relevant_docs = self.optimize_memory_usage(query, relevant_docs)
        
        if not relevant_docs:
            return (
                "I couldn't find specific information in the knowledge base to answer your question. "
                "Please try rephrasing your query or providing more details about what "
                "specific business aspect you're interested in."
            )
        
        # Prepare context for the LLM with document metadata
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            # Add document metadata and content
            doc_title = doc.get('document_title', f"Document {i+1}")
            doc_type = doc.get('document_type', 'general')
            
            context_parts.append(
                f"[{doc_title} - {doc_type}]\n{doc['chunk_text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Add any additional strategic context if provided
        if additional_context:
            context = f"{additional_context}\n\n{context}"
        
        # Prepare the prompt for the LLM
        system_prompt = """
        You are the QmiracTM AI assistant, an expert in business strategy and analysis.
        Your job is to provide accurate, helpful information based on the context provided.
        Focus on delivering strategic insights and recommendations based on the data.
        When you don't have enough information, acknowledge the limitations and suggest 
        what additional information would be helpful.
        
        Keep your answers clear, concise, and actionable. Use bullet points and structured 
        formatting when it would improve readability. If appropriate, include next steps 
        or suggestions for further analysis.
        """
        
        prompt = f"""
        Based on the following context and your business strategy expertise, please answer the query.

        Context information:
        {context}

        User query: {query}

        Provide a well-structured, strategic analysis that addresses the query directly. 
        Include relevant insights from the context and suggest practical next steps if appropriate.
        """
        
        # Generate response using the LLM
        response = self.llm.generate_response(prompt, system_prompt)
        
        # Store the interaction for learning
        self.db.store_feedback(query, response)
        
        total_time = time.time() - start_time
        print(f"Query processed in {total_time:.2f} seconds")
        
        return response
    
    def generate_strategy_recommendation(self, strategic_inputs):
        """
        Generate a comprehensive strategy recommendation.
        
        This method pulls together strategic assessment data and user inputs
        to create a complete business strategy document.
        
        Args:
            strategic_inputs: Dictionary containing strategic parameters
                - risk_tolerance: High/Medium/Low risk tolerance
                - strategic_priorities: Key priorities for the business
                - strategic_constraints: Limitations to consider
                - execution_priorities: Operational focus areas
                - execution_constraints: Operational limitations
                
        Returns:
            A comprehensive strategy recommendation document
        """
        print("Generating strategy recommendation...")
        start_time = time.time()
        
        # Store the strategic inputs in the database
        risk_tolerance = strategic_inputs.get('risk_tolerance', 'Medium')
        strategic_priorities = strategic_inputs.get('strategic_priorities', '')
        strategic_constraints = strategic_inputs.get('strategic_constraints', '')
        execution_priorities = strategic_inputs.get('execution_priorities', '')
        execution_constraints = strategic_inputs.get('execution_constraints', '')
        
        self.db.store_strategic_input(
            risk_tolerance, 
            strategic_priorities,
            strategic_constraints,
            execution_priorities,
            execution_constraints
        )
        
        # Retrieve strategy documents
        strategy_docs = self.retriever.get_relevant_documents(
            "strategic assessment market position SWOT business strategy", 
            top_k=5
        )
        
        # Retrieve execution documents
        execution_docs = self.retriever.get_relevant_documents(
            "execution operations finance marketing sales", 
            top_k=5
        )
        
        # Combine all documents
        all_docs = strategy_docs + execution_docs
        
        # Get just the most relevant docs
        all_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        context_docs = all_docs[:10]
        
        # Apply memory optimization to context docs if necessary
        context_docs = self.optimize_memory_usage("strategy generation", context_docs)
        
        # Format context with document information
        context_parts = []
        for i, doc in enumerate(context_docs):
            doc_title = doc.get('document_title', f"Document {i+1}")
            doc_type = doc.get('document_type', 'general')
            
            context_parts.append(
                f"[{doc_title} - {doc_type}]\n{doc['chunk_text']}"
            )
        
        context = "\n\n".join(context_parts)
        
        # System prompt for strategy generation
        system_prompt = """
        You are the QmiracTM Strategic Recommendation Engine, a sophisticated AI system for business strategy development.
        Your task is to synthesize information from strategic assessments and user inputs to generate a comprehensive
        strategy recommendation. Structure your response in clear sections covering strategic direction, key initiatives,
        risk assessment, implementation roadmap, and success metrics. Base your recommendations on the actual data provided
        and avoid making unfounded assumptions. Your output should be professional, insightful, and actionable.
        
        Consider the company's risk tolerance level carefully when making recommendations. For high risk tolerance,
        recommend more aggressive strategies with higher potential returns. For low risk tolerance, focus on more
        conservative approaches with steady, reliable outcomes. For medium risk tolerance, balance opportunity and caution.
        """
        
        prompt = f"""
        Based on the following strategic assessment data and user inputs, develop a comprehensive 
        strategy recommendation for the business.

        ### STRATEGIC ASSESSMENT DATA ###
        {context}

        ### USER STRATEGIC INPUTS ###
        Risk Tolerance: {risk_tolerance}
        Strategic Priorities: {strategic_priorities}
        Strategic Constraints: {strategic_constraints}
        Execution Priorities: {execution_priorities}
        Execution Constraints: {execution_constraints}

        Generate a well-structured strategy recommendation with the following sections:
        1. Executive Summary
           - Brief overview of the current situation and key recommendations
        
        2. Strategic Direction
           - Mission and vision alignment
           - Core strategic positioning
           - Key differentiators and value proposition
        
        3. Key Strategic Initiatives
           - Prioritized list of 3-5 key initiatives
           - Resources required for each initiative
           - Expected outcomes and timeline
        
        4. Risk Assessment and Mitigation
           - Analysis of major risks based on the provided risk tolerance
           - Specific mitigation strategies for each identified risk
           - Contingency planning recommendations
        
        5. Implementation Roadmap
           - Phased implementation plan with key milestones
           - Critical dependencies and sequencing
           - Resource allocation recommendations
        
        6. Critical Success Factors and KPIs
           - Specific, measurable outcomes to track
           - Monitoring and evaluation framework
           - Suggested review and adjustment process
        """
        
        # Generate the strategy recommendation with higher-quality settings
        recommendation = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            model="deepseek-coder:reasoning",  # Use reasoning-optimized model
            temperature=0.5,                   # Lower temperature for more focused output
            max_tokens=4096                    # More tokens for comprehensive strategy
        )
        
        total_time = time.time() - start_time
        print(f"Strategy recommendation generated in {total_time:.2f} seconds")
        
        return recommendation