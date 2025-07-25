import os
import re
import yaml
from typing import Tuple, Dict, Any, List
from utilities.logger import setup_logger
import time 
import json 

logger = setup_logger(__name__)

def read_markdown_file(file_path: str) -> Tuple[Dict[str, Any], str]:
    """Reads markdown file with optional YAML frontmatter.
    
    Args:
        file_path: Path to markdown file
        
    Returns:
        Tuple containing:
        - Dictionary of parsed YAML metadata (empty dict if none)
        - String of markdown content (without frontmatter)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: For malformed frontmatter
    """

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
    """Splits markdown content into logical sections.
    
    Args:
        content: Raw markdown text
        
    Returns:
        List of section strings split by headers
        
    Notes:
        - Uses '#', '##', etc. as section delimiters
        - Filters out empty sections and "NO_CONTENT_HERE" markers
    """

    # Split by headers (# or ## or ###)
    sections = re.split(r'(?=^#+ )', content, flags=re.MULTILINE)
    # Filter out empty sections and sections with just "NO_CONTENT_HERE"
    return [section.strip() for section in sections if section.strip() and "NO_CONTENT_HERE" not in section]

def extract_facts_from_section(section: str) -> List[str]:
    """Extracts discrete facts from a markdown section.
    
    Args:
        section: Markdown content (may include header)
        
    Returns:
        List of fact strings with:
        - List items expanded
        - Sentences separated
        - Minimum 10 character length
        
    Notes:
        - Preserves hierarchical list context
        - Falls back to full content if no facts extracted
    """

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
    """Processes markdown lists into structured facts.
    
    Args:
        list_text: Markdown list content
        
    Returns:
        List of processed items with:
        - Sub-items linked to parents
        - Continuation lines merged
        
    Example:
        Input: "1. Main point\n   - Subpoint"
        Output: ["Main point", "Main point - Subpoint"]
    """

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
    """Advanced sentence splitting with edge case handling.
    
    Args:
        text: Paragraph to split
        
    Returns:
        List of sentences with:
        - Abbreviations protected (e.g., "U.S.")
        - Decimal numbers preserved
        - Continuation lines merged
        
    Notes:
        Uses PROTECTED marker internally during processing
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
    """Derives clean document name from filename.
    
    Args:
        filename: Input filename (e.g., "doc.md")
        
    Returns:
        Basename without extension (e.g., "doc")
    """

    # Remove .md extension and use as document name
    return os.path.splitext(filename)[0]

def save_output_to_file(data: List[Dict[str, Any]], output_dir: str, prefix: str) -> str:
    """Saves JSON data with timestamped filename.
    
    Args:
        data: List of QA pairs to save
        output_dir: Target directory
        prefix: Filename prefix (e.g., "train")
        
    Returns:
        Absolute path to created file
        
    Notes:
        - Creates output_dir if needed
        - Uses format: "{prefix}-synthetic-data-{timestamp}.json"
    """
    
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_dir, f'{prefix}-synthetic-data-{timestamp}.json')
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    return output_path