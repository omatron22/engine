# src/llm_integration.py
"""
LLM Integration Module for QmiracTM AI-Driven Knowledge Base

This module enhances the RAG system with specialized prompting
and integration with the system prompts module.
"""

import time
from typing import Dict, List, Any, Optional
from src.system_prompts import get_system_prompt, determine_query_type

class EnhancedLLMIntegration:
    def __init__(self, llm_manager, retriever):
        """
        Initialize the enhanced LLM integration.
        
        Args:
            llm_manager: LLM Manager instance for generating responses
            retriever: Retriever instance for finding relevant documents
        """
        self.llm = llm_manager
        self.retriever = retriever
        
    def process_query(self, query: str, additional_context: Optional[str] = None):
        """
        Process a query with enhanced context handling and specialized prompting.
        
        Args:
            query: The user's question or request
            additional_context: Optional additional context to include
            
        Returns:
            Generated response from the AI
        """
        print(f"Processing query with enhanced integration: '{query}'")
        start_time = time.time()
        
        # Determine query type for specialized handling
        query_type = determine_query_type(query)
        print(f"Detected query type: {query_type}")
        
        # Get appropriate system prompt
        system_prompt = get_system_prompt(query_type)
        
        # Retrieve relevant documents with potentially adjusted parameters
        if query_type == "market" or query_type == "competitive":
            # These query types benefit from more diverse document retrieval
            relevant_docs = self.retriever.get_relevant_documents(
                query, 
                top_k=8,  # Retrieve more documents
                similarity_threshold=0.25  # Lower threshold for more diverse results
            )
        else:
            # Standard retrieval for other query types
            relevant_docs = self.retriever.get_relevant_documents(query)
        
        if not relevant_docs:
            fallback_message = (
                "I couldn't find specific information in the knowledge base to answer your question. "
                "Please try rephrasing your query or providing more details about what "
                "specific business aspect you're interested in."
            )
            
            # Add query type-specific suggestions
            if query_type == "market":
                fallback_message += (
                    "\n\nFor market assessment queries, consider specifying the market segment, "
                    "industry, or particular aspect of market analysis you're interested in."
                )
            elif query_type == "strategy":
                fallback_message += (
                    "\n\nFor strategy generation, you may want to try the 'strategy' command "
                    "which will guide you through the full strategy recommendation process."
                )
                
            return fallback_message
        
        # Prepare context for the LLM with improved document handling
        context_parts = []
        
        # Group documents by type for better context organization
        doc_groups = {}
        for doc in relevant_docs:
            doc_type = doc.get('document_type', 'general')
            if doc_type not in doc_groups:
                doc_groups[doc_type] = []
            doc_groups[doc_type].append(doc)
        
        # Add context with document type grouping
        for doc_type, docs in doc_groups.items():
            context_parts.append(f"\n## {doc_type.upper()} DOCUMENTS:")
            for i, doc in enumerate(docs):
                doc_title = doc.get('document_title', f"Document {i+1}")
                context_parts.append(
                    f"[{doc_title}]\n{doc['chunk_text']}\n"
                )
        
        context = "\n".join(context_parts)
        
        # Add any additional strategic context if provided
        if additional_context:
            context = f"{additional_context}\n\n{context}"
        
        # Prepare an enhanced prompt with query type-specific instructions
        if query_type == "market":
            prompt = self._create_market_assessment_prompt(query, context)
        elif query_type == "competitive":
            prompt = self._create_competitive_assessment_prompt(query, context)
        elif query_type == "swot":
            prompt = self._create_swot_analysis_prompt(query, context)
        elif query_type == "risk":
            prompt = self._create_risk_assessment_prompt(query, context)
        elif query_type == "finance":
            prompt = self._create_finance_analysis_prompt(query, context)
        elif query_type == "execution":
            prompt = self._create_execution_prompt(query, context)
        else:
            # Default prompt for general queries
            prompt = f"""
            Based on the following context and your business strategy expertise, please answer the query.

            Context information:
            {context}

            User query: {query}

            Provide a well-structured, strategic analysis that addresses the query directly. 
            Include relevant insights from the context and suggest practical next steps if appropriate.
            """
        
        # Generate response using the LLM with type-specific temperature
        if query_type == "strategy":
            # Lower temperature for more consistent strategy recommendations
            temperature = 0.5
        elif query_type in ["market", "competitive", "risk"]:
            # Medium temperature for analytical responses
            temperature = 0.7
        else:
            # Higher temperature for more creative general responses
            temperature = 0.8
            
        response = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        total_time = time.time() - start_time
        print(f"Query processed in {total_time:.2f} seconds")
        
        return response
    
    def _create_market_assessment_prompt(self, query, context):
        """Create specialized prompt for market assessment queries."""
        return f"""
        Based on the following business data and context, please provide a 
        comprehensive market assessment analysis addressing the query.

        Context information:
        {context}

        User query: {query}

        In your analysis, please:
        1. Identify key market factors from the provided data
        2. Assess market attractiveness based on these factors
        3. Suggest potential target segments if relevant
        4. Recommend next steps for market assessment or entry strategy
        5. Highlight any critical market trends or factors to monitor
        
        Structure your response clearly with distinct sections for market factors,
        assessment, recommendations, and next steps.
        """
    
    def _create_competitive_assessment_prompt(self, query, context):
        """Create specialized prompt for competitive assessment queries."""
        return f"""
        Based on the following business data and context, please provide a 
        detailed competitive assessment addressing the query.

        Context information:
        {context}

        User query: {query}

        In your analysis, please:
        1. Identify the key competitive metrics from the provided data
        2. Assess the business's position relative to competitors on these metrics
        3. Highlight key competitive advantages and disadvantages
        4. Suggest strategies to improve competitive positioning
        5. Recommend a framework for ongoing competitive monitoring
        
        Structure your response with clear sections and emphasize actionable insights
        derived from the specific competitive data available.
        """
    
    def _create_swot_analysis_prompt(self, query, context):
        """Create specialized prompt for SWOT analysis queries."""
        return f"""
        Based on the following business data and context, please provide a
        structured SWOT analysis addressing the query.

        Context information:
        {context}

        User query: {query}

        In your analysis, please:
        1. Identify key strengths, weaknesses, opportunities, and threats
        2. Prioritize the most significant factors in each category
        3. Analyze connections between SWOT elements (e.g., how strengths can address threats)
        4. Suggest strategic initiatives based on the SWOT analysis
        5. Recommend an approach for ongoing SWOT monitoring and updates
        
        Format your response with clear sections for each SWOT component and ensure
        recommendations are specific and actionable rather than generic.
        """
    
    def _create_risk_assessment_prompt(self, query, context):
        """Create specialized prompt for risk assessment queries."""
        return f"""
        Based on the following business data and context, please provide a
        comprehensive risk assessment addressing the query.

        Context information:
        {context}

        User query: {query}

        In your analysis, please:
        1. Identify and categorize key business risks from the provided data
        2. Assess probability and potential impact of each risk
        3. Suggest appropriate risk mitigation strategies
        4. Recommend a risk monitoring approach
        5. Consider both downside risks and potential upside opportunities
        
        Structure your response with clear risk categories and ensure recommendations
        are practical and aligned with business capabilities.
        """
    
    def _create_finance_analysis_prompt(self, query, context):
        """Create specialized prompt for financial analysis queries."""
        return f"""
        Based on the following financial data and business context, please provide a
        detailed financial analysis addressing the query.

        Context information:
        {context}

        User query: {query}

        In your analysis, please:
        1. Interpret key financial metrics and trends from the provided data
        2. Connect financial performance to strategic decisions
        3. Identify areas of financial strength and concern
        4. Suggest strategies for financial improvement
        5. Recommend key financial metrics to monitor going forward
        
        Structure your response clearly and ensure recommendations are grounded
        in the actual financial data provided.
        """
    
    def _create_execution_prompt(self, query, context):
        """Create specialized prompt for execution planning queries."""
        return f"""
        Based on the following business data and context, please provide
        execution planning guidance addressing the query.

        Context information:
        {context}

        User query: {query}

        In your response, please:
        1. Outline a structured approach to execution planning
        2. Suggest how to translate strategic objectives into actionable tasks
        3. Recommend appropriate metrics and KPIs for tracking progress
        4. Address resource allocation considerations
        5. Highlight potential execution challenges and mitigation approaches
        
        Structure your response with clear, practical steps and ensure recommendations
        are specific and implementable rather than theoretical.
        """
    
    def generate_strategy_recommendation(self, strategic_inputs):
        """
        Generate a comprehensive strategy recommendation with enhanced structure and content.
        
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
        print("Generating enhanced strategy recommendation...")
        start_time = time.time()
        
        # Extract strategic input parameters
        risk_tolerance = strategic_inputs.get('risk_tolerance', 'Medium')
        strategic_priorities = strategic_inputs.get('strategic_priorities', '')
        strategic_constraints = strategic_inputs.get('strategic_constraints', '')
        execution_priorities = strategic_inputs.get('execution_priorities', '')
        execution_constraints = strategic_inputs.get('execution_constraints', '')
        
        # Get the specialized strategy generation system prompt
        system_prompt = get_system_prompt("strategy")
        
        # Retrieve relevant documents for strategy formulation with broader coverage
        strategy_docs = self.retriever.get_relevant_documents(
            "strategic assessment market position SWOT business strategy", 
            top_k=6,
            similarity_threshold=0.25  # Lower threshold for broader coverage
        )
        
        execution_docs = self.retriever.get_relevant_documents(
            "execution operations finance marketing sales", 
            top_k=6,
            similarity_threshold=0.25
        )
        
        # Combine and prioritize documents
        all_docs = strategy_docs + execution_docs
        all_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        context_docs = all_docs[:12]  # Use more documents for comprehensive strategy
        
        # Format context with improved document organization
        context_parts = []
        
        # Group documents by type
        doc_groups = {}
        for doc in context_docs:
            doc_type = doc.get('document_type', 'general')
            if doc_type not in doc_groups:
                doc_groups[doc_type] = []
            doc_groups[doc_type].append(doc)
        
        # Add context with document type grouping
        for doc_type, docs in doc_groups.items():
            context_parts.append(f"\n## {doc_type.upper()} INFORMATION:")
            for i, doc in enumerate(docs):
                doc_title = doc.get('document_title', f"Document {i+1}")
                context_parts.append(
                    f"[{doc_title}]\n{doc['chunk_text']}\n"
                )
        
        context = "\n".join(context_parts)
        
        # Enhanced strategy generation prompt
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
        
        # Executive Summary
        - Brief overview of current situation
        - Summary of key strategic recommendations
        - Expected outcomes and timeframe
        
        # Strategic Direction
        - Mission and vision alignment
        - Core strategic positioning
        - Key differentiators and value proposition
        - Target market focus
        
        # Key Strategic Initiatives
        - Prioritized list of 3-5 key initiatives
        - Resources required for each initiative
        - Expected outcomes and timeline
        - Success metrics for each initiative
        
        # Risk Assessment and Mitigation
        - Analysis of major risks based on the provided risk tolerance ({risk_tolerance})
        - Specific mitigation strategies for each identified risk
        - Contingency planning recommendations
        - Opportunity assessment within risk profile
        
        # Implementation Roadmap
        - Phased implementation plan with key milestones
        - Critical dependencies and sequencing
        - Resource allocation recommendations
        - Quick wins vs. long-term initiatives
        
        # Critical Success Factors and KPIs
        - Specific, measurable outcomes to track
        - Monitoring and evaluation framework
        - Suggested review and adjustment process
        - Leading vs. lagging indicators
        
        Ensure your recommendations directly address the strategic priorities and are realistic
        given the stated constraints. Tailor your strategy to the specified risk tolerance level.
        """
        
        # Generate the strategy recommendation with specialized settings
        recommendation = self.llm.generate_response(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.4,  # Lower temperature for more focused strategy output
            max_tokens=4096   # Allow for a comprehensive strategy document
        )
        
        total_time = time.time() - start_time
        print(f"Enhanced strategy recommendation generated in {total_time:.2f} seconds")
        
        return recommendation