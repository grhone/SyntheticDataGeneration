import os
import re
import yaml
from typing import Tuple, Dict, Any, List
from utilities.logger import setup_logger
import time 
import json 

logger = setup_logger(__name__)

def read_markdown_file(file_path: str) -> Tuple[Dict[str, Any], str]:
    """Read content from a markdown file and extract metadata."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Extract YAML frontmatter if present
        metadata = {}
        if content.startswith('---'):
            try:
                frontmatter_end = content.index('---', 3)
                yaml_content = content[3:frontmatter_end].strip()
                metadata = yaml.safe_load(yaml_content) or {}
                content = content[frontmatter_end+3:].lstrip()
            except Exception as e:
                logger.warning(f"YAML parsing failed for {file_path}: {e}")
                
        return metadata, content

def split_into_sections(content: str) -> List[str]:
    """Split the markdown content into meaningful sections."""
    # Split by headers (# or ## or ###)
    sections = re.split(r'(?=^#+ )', content, flags=re.MULTILINE)
    # Filter out empty sections and sections with just "NO_CONTENT_HERE"
    return [section.strip() for section in sections if section.strip() and "NO_CONTENT_HERE" not in section]

def extract_facts_from_section(section: str) -> List[str]:
    """Extract individual facts from a section."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # List to store extracted facts
    facts = []
    
    # Split content into paragraphs
    paragraphs = re.split(r'\n\s*\n', content)
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Check if this is a numbered or bulleted list
        if re.match(r'^\s*(\d+\.|\*|\-)\s', paragraph):
            # Process list items
            list_items = process_list_items(paragraph)
            facts.extend(list_items)
        else:
            # Process regular paragraph
            sentences = split_into_sentences(paragraph)
            facts.extend(sentences)
    
    # Filter out empty facts and very short ones (likely not complete facts)
    facts = [fact.strip() for fact in facts if len(fact.strip()) > 10]
    
    # If no facts were extracted (maybe the section was too short), use the whole content as one fact
    if not facts and content.strip():
        facts = [content.strip()]
    
    return facts

def process_list_items(list_text: str) -> List[str]:
    """Process a list into individual fact items."""
    facts = []
    
    # Split the list text into lines
    lines = list_text.split('\n')
    
    # Current parent item for context
    current_parent = ""
    parent_indent = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Calculate indentation level
        indent = len(line) - len(line.lstrip())
        
        # Check if this is a list item
        list_item_match = re.match(r'^\s*(\d+\.|\*|\-)\s+(.*)', line)
        if list_item_match:
            marker, item_text = list_item_match.groups()
            
            # If this is a sub-item, include the parent context
            if indent > parent_indent:
                fact = f"{current_parent} - {item_text}"
            else:
                # This is a top-level item or a new sub-list
                fact = item_text
                current_parent = item_text
                parent_indent = indent
                
            facts.append(fact)
        else:
            # This is continuation text for the previous item
            if facts:
                facts[-1] += " " + line
    
    return facts

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences, handling abbreviations and special cases.
    """
    # Protect common abbreviations from being split
    protected_text = text
    protected_text = re.sub(r'(\w\.\w\.)', r'\1PROTECTED', protected_text)  # Protect abbreviations like U.S.
    protected_text = re.sub(r'(\d+\.\d+)', r'\1PROTECTED', protected_text)  # Protect decimal numbers
    protected_text = re.sub(r'([A-Za-z]\.[A-Za-z]\.)', r'\1PROTECTED', protected_text)  # Protect Pa.C.S.
    
    # Split on sentence boundaries
    sentence_boundaries = r'(?<=[.!?])\s+'
    raw_sentences = re.split(sentence_boundaries, protected_text)
    
    # Restore protected text
    sentences = [re.sub(r'PROTECTED', '', s) for s in raw_sentences]
    
    # Further process sentences to handle special cases
    processed_sentences = []
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        # Handle sentences that might have been incorrectly split
        if re.match(r'^[a-z]', sentence) and processed_sentences:
            # This sentence starts with lowercase, likely continuation of previous
            processed_sentences[-1] += " " + sentence
        else:
            processed_sentences.append(sentence)
    
    return processed_sentences

def get_document_name_from_filename(filename: str) -> str:
    """Extract document name from markdown filename."""
    # Remove .md extension and use as document name
    return os.path.splitext(filename)[0]

def save_output_to_file(data: List[Dict[str, Any]], output_dir: str, prefix: str) -> str:
    """Save processed data to a JSON file with timestamp."""
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f'{prefix}-synthetic-data-{timestamp}.json')
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_path