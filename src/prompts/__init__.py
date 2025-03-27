"""
QmiracTM AI-Driven Knowledge Base
=================================

Prompt package for specialized system prompts.
"""

# Make functions available at the package level
from .system_prompts import (
    get_system_prompt,
    determine_query_type,
    get_prompt_temperature
)

__all__ = ['get_system_prompt', 'determine_query_type', 'get_prompt_temperature']