#!/usr/bin/env python3
"""
QmiracTM PDF Strategy Generator

Generates professional PDF strategy recommendation documents
from the RAG system output.
"""
import os
import time
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
import re

class StrategyOutputGenerator:
    def __init__(self, output_dir="strategy_outputs"):
        """
        Initialize the strategy output generator.
        
        Args:
            output_dir: Directory where strategy PDFs will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Set up custom paragraph styles for the PDF document."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='QmiracTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.HexColor('#003366')
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='QmiracSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            alignment=1,  # Center
            textColor=colors.HexColor('#666666')
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceBefore=15,
            spaceAfter=10,
            textColor=colors.HexColor('#003366'),
            borderWidth=1,
            borderColor=colors.HexColor('#003366'),
            borderPadding=5,
            borderRadius=5
        ))
        
        # Normal text style
        self.styles.add(ParagraphStyle(
            name='QmiracNormal',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=10,
            leading=14
        ))
        
        # List item style
        self.styles.add(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=3,
            leftIndent=20,
            leading=14
        ))
        
    def _parse_strategy_sections(self, text):
        """Parse the strategy text into structured sections."""
        # Define regex patterns for section headers based on common formats in the output
        section_patterns = [
            r'^#+\s+(.+)$',                      # Markdown headers: # Header
            r'^(\d+\.\s+.+)$',                   # Numbered sections: 1. Executive Summary
            r'^([A-Z][A-Za-z\s]+):$',            # Title with colon: Executive Summary:
            r'^([A-Z][A-Za-z\s]+)$'              # All caps or title case: EXECUTIVE SUMMARY or Executive Summary
        ]
        
        # Find all sections
        sections = []
        current_section = {"title": "Overview", "content": ""}
        
        # Split by common section headers
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                current_section["content"] += "\n\n"
                continue
                
            match = False
            for pattern in section_patterns:
                if re.match(pattern, line, re.MULTILINE):
                    # If we found a section header and we have content in the current section,
                    # save the current section and start a new one
                    if current_section["content"]:
                        sections.append(current_section)
                    
                    # Create a new section
                    current_section = {
                        "title": re.sub(r'^#+\s+|\d+\.\s+|:\s*$', '', line).strip(),
                        "content": ""
                    }
                    match = True
                    break
            
            if not match:
                current_section["content"] += line + "\n"
        
        # Add the last section
        if current_section["content"]:
            sections.append(current_section)
            
        return sections
    
    def generate_pdf(self, strategy_text, strategic_inputs, filename=None):
        """
        Generate a PDF document from strategy text.
        
        Args:
            strategy_text: The strategy recommendation text
            strategic_inputs: Dictionary containing strategic parameters
            filename: Optional filename for the PDF (default: auto-generated)
            
        Returns:
            Path to the generated PDF
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"QmiracTM_Strategy_{timestamp}.pdf"
        
        full_path = os.path.join(self.output_dir, filename)
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            full_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Parse the strategy text into sections
        sections = self._parse_strategy_sections(strategy_text)
        
        # Build the PDF content
        content = []
        
        # Add title
        content.append(Paragraph("QmiracTM Strategy Recommendation", self.styles["QmiracTitle"]))
        
        # Add date
        date_str = datetime.now().strftime("%B %d, %Y")
        content.append(Paragraph(f"Generated on {date_str}", self.styles["QmiracSubtitle"]))
        content.append(Spacer(1, 0.25 * inch))
        
        # Add strategic inputs table
        content.append(Paragraph("Strategic Inputs", self.styles["SectionHeader"]))
        content.append(Spacer(1, 0.1 * inch))
        
        # Create table data
        table_data = [
            ["Parameter", "Value"],
            ["Risk Tolerance", strategic_inputs.get('risk_tolerance', 'Medium')],
            ["Strategic Priorities", strategic_inputs.get('strategic_priorities', 'N/A')],
            ["Strategic Constraints", strategic_inputs.get('strategic_constraints', 'N/A')],
            ["Execution Priorities", strategic_inputs.get('execution_priorities', 'N/A')],
            ["Execution Constraints", strategic_inputs.get('execution_constraints', 'N/A')]
        ]
        
        # Create table
        table = Table(table_data, colWidths=[2*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor('#003366')),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.white),
            ('ALIGN', (0, 0), (1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (1, 0), 12),
            ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#E5E5E5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        content.append(table)
        content.append(Spacer(1, 0.25 * inch))
        
        # Add strategy sections
        for section in sections:
            title = section["title"]
            section_content = section["content"].strip()
            
            content.append(Paragraph(title, self.styles["SectionHeader"]))
            
            # Process content: split by newlines and handle bullet points
            paragraphs = section_content.split('\n')
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # Check if it's a bullet point
                if para.startswith('-') or para.startswith('•') or re.match(r'^\d+\.', para):
                    style = self.styles["ListItem"]
                    # Ensure proper indentation for bullet points
                    if para.startswith('-') or para.startswith('•'):
                        para = para[1:].strip()
                        para = f"• {para}"
                else:
                    style = self.styles["QmiracNormal"]
                    
                content.append(Paragraph(para, style))
                
            content.append(Spacer(1, 0.1 * inch))
        
        # Add footer with disclaimer
        content.append(Spacer(1, 0.5 * inch))
        disclaimer = (
            "DISCLAIMER: This strategy recommendation was generated by QmiracTM AI-Driven Knowledge Base based on "
            "the provided business data and inputs. This document is intended to assist in strategic decision-making "
            "but should be reviewed and validated by business experts before implementation."
        )
        content.append(Paragraph(disclaimer, ParagraphStyle(
            name='Disclaimer',
            parent=self.styles['Normal'],
            fontSize=8,
            textColor=colors.gray
        )))
        
        # Build the PDF
        doc.build(content)
        
        return full_path