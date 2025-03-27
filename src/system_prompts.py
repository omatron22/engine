# src/system_prompts.py
"""
System Prompts Module for QmiracTM AI-Driven Knowledge Base

This module contains specialized system prompts and utilities
for different types of business strategy queries.
"""

# Import the system prompts from a centralized location
from src.prompts.system_prompts import (
    GENERAL_QUERY_PROMPT,
    STRATEGY_GENERATION_PROMPT,
    MARKET_ASSESSMENT_PROMPT,
    COMPETITIVE_ASSESSMENT_PROMPT,
    SWOT_ANALYSIS_PROMPT,
    RISK_ASSESSMENT_PROMPT,
    EXECUTION_PLANNING_PROMPT,
    FINANCE_DASHBOARD_PROMPT
)

def get_system_prompt(query_type="general"):
    """
    Return the appropriate system prompt based on query type.
    
    Args:
        query_type: Type of query (general, strategy, market, etc.)
        
    Returns:
        Appropriate system prompt string
    """
    prompts = {
        "general": GENERAL_QUERY_PROMPT,
        "strategy": STRATEGY_GENERATION_PROMPT,
        "market": MARKET_ASSESSMENT_PROMPT,
        "competitive": COMPETITIVE_ASSESSMENT_PROMPT,
        "swot": SWOT_ANALYSIS_PROMPT,
        "risk": RISK_ASSESSMENT_PROMPT,
        "execution": EXECUTION_PLANNING_PROMPT,
        "finance": FINANCE_DASHBOARD_PROMPT
    }
    
    return prompts.get(query_type.lower(), GENERAL_QUERY_PROMPT)

def determine_query_type(query_text):
    """
    Attempt to determine the query type from the query text.
    
    Args:
        query_text: The user's query
        
    Returns:
        Detected query type (general, strategy, market, etc.)
    """
    query_text = query_text.lower()
    
    # Strategy generation
    if any(term in query_text for term in ["strategy recommendation", "strategic plan", "generate strategy"]):
        return "strategy"
    
    # Market assessment
    elif any(term in query_text for term in ["market", "customer segment", "target market"]):
        return "market"
    
    # Competitive assessment
    elif any(term in query_text for term in ["compet", "rival", "industry player", "market share"]):
        return "competitive"
    
    # SWOT analysis
    elif any(term in query_text for term in ["swot", "strength", "weakness", "opportunity", "threat"]):
        return "swot"
    
    # Risk assessment
    elif any(term in query_text for term in ["risk", "uncertainty", "threat", "mitigation"]):
        return "risk"
    
    # Execution planning
    elif any(term in query_text for term in ["execution", "implement", "operation", "tactical"]):
        return "execution"
    
    # Financial analysis
    elif any(term in query_text for term in ["financ", "revenue", "profit", "cost", "budget"]):
        return "finance"
    
    # Default to general
    else:
        return "general"

def get_prompt_temperature(query_type):
    """
    Get the recommended temperature setting for a query type.
    
    Args:
        query_type: Type of query
        
    Returns:
        Recommended temperature value (0.0-1.0)
    """
    temperatures = {
        "strategy": 0.4,    # More focused for strategy generation
        "market": 0.6,      # Balanced for market assessment
        "competitive": 0.6, # Balanced for competitive assessment
        "swot": 0.5,        # Moderately focused for structured analysis
        "risk": 0.5,        # Moderately focused for risk assessment
        "execution": 0.7,   # More creative for implementation ideas
        "finance": 0.4,     # More focused for financial analysis
        "general": 0.7      # More creative for general queries
    }
    
    return temperatures.get(query_type.lower(), 0.7)