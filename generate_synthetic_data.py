import os
import json
import random
import re
import time
from typing import List, Dict, Any, Tuple, Optional
import argparse
import nest_asyncio
from enum import Enum, auto

# Configuration
MARKDOWN_DOCS_DIR = "markdown_docs"
OUTPUT_DIR = "output"
TRAIN_RATIO = 0.9  # 90% for training, 10% for evaluation

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define question types
class QuestionType(Enum):
    FACTUAL_RECALL = auto()
    INFERENCE = auto()
    MULTI_HOP_REASONING = auto()
    APPLICATION = auto()
    COMPARATIVE_ANALYSIS = auto()
    CAUSE_EFFECT = auto()
    SUMMARIZATION = auto()
    HYPOTHETICAL = auto()
    CRITICAL_ANALYSIS = auto()
    TECHNICAL_EXPLANATION = auto()
    PROCESS_WORKFLOW = auto()
    TRUE_FALSE_FILL_BLANK = auto()
    CONTEXTUAL_AMBIGUITY = auto()
    
    # Original question types
    FACT_BASED = auto()
    SECTION_SUMMARY = auto()

# Define difficulty levels
class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# Question type distribution (percentages)
QUESTION_TYPE_DISTRIBUTION = {
    QuestionType.FACTUAL_RECALL: 15,
    QuestionType.INFERENCE: 10,
    QuestionType.MULTI_HOP_REASONING: 5,
    QuestionType.APPLICATION: 8,
    QuestionType.COMPARATIVE_ANALYSIS: 5,
    QuestionType.CAUSE_EFFECT: 5,
    QuestionType.SUMMARIZATION: 5,
    QuestionType.HYPOTHETICAL: 5,
    QuestionType.CRITICAL_ANALYSIS: 5,
    QuestionType.TECHNICAL_EXPLANATION: 10,
    QuestionType.PROCESS_WORKFLOW: 5,
    QuestionType.TRUE_FALSE_FILL_BLANK: 10,
    QuestionType.CONTEXTUAL_AMBIGUITY: 5,
    # Original question types are handled separately
}

# Difficulty distribution (percentages)
DIFFICULTY_DISTRIBUTION = {
    DifficultyLevel.EASY: 60,
    DifficultyLevel.MEDIUM: 30,
    DifficultyLevel.HARD: 10,
}

# Question type to difficulty mapping
QUESTION_TYPE_DIFFICULTY = {
    QuestionType.FACTUAL_RECALL: DifficultyLevel.EASY,
    QuestionType.INFERENCE: DifficultyLevel.MEDIUM,
    QuestionType.MULTI_HOP_REASONING: DifficultyLevel.HARD,
    QuestionType.APPLICATION: DifficultyLevel.MEDIUM,
    QuestionType.COMPARATIVE_ANALYSIS: DifficultyLevel.MEDIUM,
    QuestionType.CAUSE_EFFECT: DifficultyLevel.MEDIUM,
    QuestionType.SUMMARIZATION: DifficultyLevel.EASY,
    QuestionType.HYPOTHETICAL: DifficultyLevel.HARD,
    QuestionType.CRITICAL_ANALYSIS: DifficultyLevel.HARD,
    QuestionType.TECHNICAL_EXPLANATION: DifficultyLevel.MEDIUM,
    QuestionType.PROCESS_WORKFLOW: DifficultyLevel.EASY,
    QuestionType.TRUE_FALSE_FILL_BLANK: DifficultyLevel.EASY,
    QuestionType.CONTEXTUAL_AMBIGUITY: DifficultyLevel.HARD,
    QuestionType.FACT_BASED: DifficultyLevel.EASY,
    QuestionType.SECTION_SUMMARY: DifficultyLevel.MEDIUM,
}

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate synthetic training data from markdown files')
parser.add_argument('--model', type=str, default='OpenGVLab/InternVL3-8B', help='Model to use')
parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum tokens for model response')
parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model response')
parser.add_argument('--no-llm', action='store_true', help='Use simple heuristics instead of LLM')
args = parser.parse_args()

def read_markdown_file(file_path: str) -> str:
    """Read content from a markdown file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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

def get_json_str(string: str) -> str:
    """Extract JSON string from text."""
    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return string[first:last + 1]

def setup_lmdeploy_pipeline():
    """Set up and return the lmdeploy pipeline."""
    try:
        from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
        
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # System prompt for the model
        system_prompt = 'You are a knowledgeable and helpful teacher who generates detailed JSON responses for questions, integrating specific writing styles from provided text without directly referencing the original document.'
        
        # Initialize chat template configuration
        chat_template_config = ChatTemplateConfig('internvl2_5')
        chat_template_config.meta_instruction = system_prompt
        
        # Initialize backend configuration
        backend_config = TurbomindEngineConfig(tp=2)
        
        # Create the pipeline with the provided configurations
        pipe = pipeline(args.model, chat_template_config=chat_template_config, backend_config=backend_config)
        
        return pipe
    except ImportError as e:
        print(f"Error importing lmdeploy: {e}")
        print("Please install lmdeploy with: pip install lmdeploy[all]")
        return None
    except Exception as e:
        print(f"Error setting up lmdeploy pipeline: {e}")
        return None

def generate_fact_qa_pair_with_lmdeploy(pipe, fact: str, section_index: int, fact_index: int) -> Dict[str, Any]:
    """Generate a question-answer pair for a single fact using lmdeploy."""
    # Prepare the prompt
    prompt = f"""
For the following fact from a document about Highly Automated Vehicles (HAV) regulations, generate a self-contained question/answer pair. Reflect the writing style, tone, and thematic elements of the original document without directly referencing or quoting the text. Follow these steps:

1. Analyze the fact's style, including language patterns, tone, and structure.
2. Create a question that specifically targets this single fact, ensuring it is self-contained and independent of the original text.
3. Provide a detailed answer that reflects the document's style and thoroughly explains the concept.
4. Use markdown format for the answer where appropriate.

Ensure all outputs are independent of the original text context. The question and answer should appear as standalone general knowledge content. Use the following JSON format:

{{
  "section_number": "{section_index}",
  "fact_number": "{fact_index}",
  "question_type": "FACT_BASED",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the fact:

{fact}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating fact QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_fact_qa_pair_simple(fact, section_index, fact_index)

def generate_section_summary_qa_pair_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a summary question-answer pair for a section using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following section from a document about Highly Automated Vehicles (HAV) regulations, generate a summary question/answer pair. The question should ask for a comprehensive summary of the section, and the answer should summarize all key information in the section. Reflect the writing style, tone, and thematic elements of the original document without directly referencing or quoting the text. Follow these steps:

1. Analyze the section's style, including language patterns, tone, and structure.
2. Create a question that asks for a summary of the key information in this section.
3. Provide a detailed answer that summarizes all important information in the section.
4. Use markdown format for the answer where appropriate.

Ensure all outputs are independent of the original text context. The question and answer should appear as standalone general knowledge content. Use the following JSON format:

{{
  "section_number": "{section_index}",
  "is_summary": true,
  "question_type": "SECTION_SUMMARY",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating section summary QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_section_summary_qa_pair_simple(section, section_index)

def generate_qa_pair_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """
    Legacy function maintained for compatibility.
    Generate a question-answer pair with reasoning steps using lmdeploy.
    """
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following paragraph from a document about Highly Automated Vehicles (HAV) regulations, generate a self-contained question/answer pair. Reflect the writing style, tone, and thematic elements of the original document without directly referencing or quoting the text. Follow these steps:

1. Analyze the paragraph's style, including language patterns, tone, and structure.
2. Create a question based on the implicit themes or knowledge in the paragraph, ensuring it is self-contained and independent of the original text.
3. Provide a detailed answer that reflects the document's style and thoroughly explains the concept.
4. Use markdown format for the answer where appropriate.

Ensure all outputs are independent of the original text context. The question and answer should appear as standalone general knowledge content. Use the following JSON format:

{{
  "paragraph_number": "{section_index}",
  "question_type": "FACT_BASED",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the paragraph:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_qa_pair_simple(section, section_index)

def generate_fact_qa_pair_simple(fact: str, section_index: int, fact_index: int) -> Dict[str, Any]:
    """Generate a question-answer pair for a single fact using simple heuristics."""
    # Simple heuristic to generate a question based on the fact
    if "definition" in fact.lower():
        question = f"What is the definition related to Highly Automated Vehicles mentioned in this fact?"
    elif "certification" in fact.lower():
        question = f"What certification requirement is mentioned in this fact about Highly Automated Vehicles?"
    elif "safety" in fact.lower():
        question = f"What safety measure for Highly Automated Vehicles is described in this fact?"
    elif "emergency" in fact.lower():
        question = f"What emergency procedure related to Highly Automated Vehicles is mentioned in this fact?"
    elif "prohibition" in fact.lower():
        question = f"What prohibition regarding Highly Automated Vehicles is described in this fact?"
    elif "reporting" in fact.lower():
        question = f"What reporting requirement for Highly Automated Vehicles is mentioned in this fact?"
    else:
        # Extract first few words to create a question
        words = fact.split()[:3]
        topic = " ".join(words)
        question = f"What does the Pennsylvania Department of Transportation guideline say about {topic}?"
    
    # Generate an answer based on the fact
    answer = f"According to the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles: {fact}"
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the key information in this specific fact.",
        f"This fact contains important information about a specific aspect of Highly Automated Vehicles regulation.",
        f"Based on this fact, I can provide a precise answer about this specific aspect of Highly Automated Vehicles in Pennsylvania."
    ]
    
    return {
        "section_number": str(section_index),
        "fact_number": str(fact_index),
        "question_type": "FACT_BASED",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_section_summary_qa_pair_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a summary question-answer pair for a section using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Generate a summary question based on the title
    question = f"Can you summarize the key points about {title} in the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles?"
    
    # Generate a summary answer based on the content
    # For simplicity, we'll use the first 500 characters as a summary
    answer = f"The Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles provide the following key information about {title.lower()}: {content[:500]}..."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the main topic of this section, which is {title.lower()}.",
        f"Next, I need to extract the key points about {title.lower()} from the Pennsylvania guidelines.",
        f"Finally, I can summarize these key points to provide a comprehensive overview of {title.lower()} in the context of Highly Automated Vehicles in Pennsylvania."
    ]
    
    return {
        "section_number": str(section_index),
        "is_summary": True,
        "question_type": "SECTION_SUMMARY",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_qa_pair_simple(section: str, section_index: int) -> Dict[str, Any]:
    """
    Legacy function maintained for compatibility.
    Generate a question-answer pair with reasoning steps using simple heuristics.
    """
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Simple heuristic to generate a question based on the title
    if "definition" in title.lower() or "definitions" in title.lower():
        question = f"What is the definition of a term related to Highly Automated Vehicles in the Pennsylvania guidelines?"
    elif "certification" in title.lower():
        question = f"What are the requirements for certification of compliance for Highly Automated Vehicles in Pennsylvania?"
    elif "safety" in title.lower():
        question = f"What safety measures are required for Highly Automated Vehicles in Pennsylvania?"
    elif "emergency" in title.lower():
        question = f"What procedures should Emergency Service Responders follow when interacting with Highly Automated Vehicles?"
    elif "prohibition" in title.lower():
        question = f"What operations are prohibited for Highly Automated Vehicles in Pennsylvania?"
    elif "reporting" in title.lower():
        question = f"What reporting requirements exist for Certificate Holders of Highly Automated Vehicles in Pennsylvania?"
    else:
        question = f"What does the Pennsylvania Department of Transportation guideline say about {title}?"
    
    # Generate an answer based on the content
    answer = f"According to the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles, {title.lower()} involves the following: {content[:500]}..."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify what the Pennsylvania guidelines say about {title.lower()}.",
        f"The guidelines specify several key points about {title.lower()}, including regulatory requirements and procedures.",
        f"Based on these guidelines, I can provide a comprehensive answer about {title.lower()} in the context of Highly Automated Vehicles in Pennsylvania."
    ]
    
    return {
        "paragraph_number": str(section_index),
        "question_type": "FACT_BASED",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

# New functions for generating different question types

def generate_factual_recall_qa_with_lmdeploy(pipe, facts: List[str], section: str, section_index: int) -> Dict[str, Any]:
    """Generate a factual recall question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Select a random fact from the section
    fact = random.choice(facts) if facts else content[:500]
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a factual recall question that can be answered by directly quoting 1-2 sentences from the document. The question should test the reader's ability to locate and recall explicit information from the text.

Follow these steps:
1. Analyze the content to identify key facts, definitions, or statements.
2. Create a question that targets specific information that can be directly quoted from the text.
3. Provide an answer that includes exact text excerpts.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "FACTUAL_RECALL",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}

And here is a specific fact from this section that you might focus on:

{fact}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating factual recall QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_factual_recall_qa_simple(facts, section, section_index)

def generate_inference_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate an inference question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate an inference question that requires logical reasoning from the text. The question should test the reader's ability to understand implicit information and draw conclusions that are not explicitly stated.

Follow these steps:
1. Analyze the content to identify underlying principles, implications, or connections.
2. Create a question that requires connecting ideas or inferring meaning beyond what is directly stated.
3. Provide an answer that explains the logical reasoning process without directly quoting the text.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "INFERENCE",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating inference QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_inference_qa_simple(section, section_index)

def generate_multi_hop_qa_with_lmdeploy(pipe, sections: List[str], section_index: int) -> Dict[str, Any]:
    """Generate a multi-hop reasoning question-answer pair using lmdeploy."""
    # Need at least two sections for multi-hop reasoning
    if len(sections) < 2:
        return None
    
    # Select two random sections
    section1, section2 = random.sample(sections, 2)
    
    # Extract the section titles if they exist
    title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
    title1 = title_match1.group(1).strip() if title_match1 else "Section A"
    
    title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
    title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
    # Extract the content without the titles
    content1 = re.sub(r'^#+ .*?$', '', section1, count=1, flags=re.MULTILINE).strip()
    content2 = re.sub(r'^#+ .*?$', '', section2, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following two document sections about Highly Automated Vehicles (HAV) regulations, generate a multi-hop reasoning question that requires combining information from both sections. The question should test the reader's ability to connect ideas across different parts of the document.

Follow these steps:
1. Analyze both sections to identify related concepts or complementary information.
2. Create a question that requires integrating knowledge from both sections to answer correctly.
3. Provide an answer that explains how information from both sections contributes to the solution.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "MULTI_HOP_REASONING",
  "difficulty": "HARD",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here are the sections:

Section 1: {title1}
{content1}

Section 2: {title2}
{content2}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating multi-hop reasoning QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_multi_hop_qa_simple(sections, section_index)

def generate_application_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate an application/practical use question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a scenario-based question asking how to apply a concept from the document. The question should test the reader's ability to apply theoretical knowledge to practical situations.

Follow these steps:
1. Analyze the content to identify concepts, procedures, or guidelines that could be applied in real-world scenarios.
2. Create a question that presents a realistic scenario where this knowledge would need to be applied.
3. Provide a step-by-step answer that explains how to apply the concept in the given scenario.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "APPLICATION",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating application QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_application_qa_simple(section, section_index)

def generate_comparative_analysis_qa_with_lmdeploy(pipe, sections: List[str], section_index: int) -> Dict[str, Any]:
    """Generate a comparative analysis question-answer pair using lmdeploy."""
    # Need at least two sections for comparative analysis
    if len(sections) < 2:
        return None
    
    # Select two random sections
    section1, section2 = random.sample(sections, 2)
    
    # Extract the section titles if they exist
    title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
    title1 = title_match1.group(1).strip() if title_match1 else "Section A"
    
    title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
    title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
    # Extract the content without the titles
    content1 = re.sub(r'^#+ .*?$', '', section1, count=1, flags=re.MULTILINE).strip()
    content2 = re.sub(r'^#+ .*?$', '', section2, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following two document sections about Highly Automated Vehicles (HAV) regulations, generate a comparative analysis question that asks the reader to compare two concepts or approaches. The question should test the reader's ability to identify similarities and differences between related ideas.

Follow these steps:
1. Analyze both sections to identify concepts, approaches, or requirements that can be meaningfully compared.
2. Create a question that asks for a comparison, highlighting specific aspects to focus on.
3. Provide an answer that systematically analyzes similarities and differences.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "COMPARATIVE_ANALYSIS",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here are the sections:

Section 1: {title1}
{content1}

Section 2: {title2}
{content2}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating comparative analysis QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_comparative_analysis_qa_simple(sections, section_index)

def generate_cause_effect_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a cause-effect relationship question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a question about cause-effect chains described in the document. The question should test the reader's ability to understand causal relationships and their implications.

Follow these steps:
1. Analyze the content to identify cause-effect relationships, where one event or condition leads to specific outcomes.
2. Create a question that asks about these causal relationships, focusing on why certain effects occur.
3. Provide an answer that explains the causal chain, citing specific parts of the document.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "CAUSE_EFFECT",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating cause-effect QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_cause_effect_qa_simple(section, section_index)

def generate_summarization_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a summarization question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a summarization question that asks for a concise summary of multi-page content. The question should test the reader's ability to identify and synthesize key information.

Follow these steps:
1. Analyze the content to identify the main topics, key points, and essential information.
2. Create a question that asks for a concise summary of specific aspects of the content.
3. Provide an answer that summarizes the requested information in under 100 words.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "SUMMARIZATION",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating summarization QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_summarization_qa_simple(section, section_index)

def generate_hypothetical_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a hypothetical scenario question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a 'what-if' question that extends document concepts to new situations. The question should test the reader's ability to apply principles from the document to hypothetical scenarios.

Follow these steps:
1. Analyze the content to identify key principles, rules, or guidelines that could be applied to new situations.
2. Create a hypothetical scenario that extends beyond what is explicitly covered in the document.
3. Provide a plausible speculative answer based on the principles established in the document.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "HYPOTHETICAL",
  "difficulty": "HARD",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating hypothetical QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_hypothetical_qa_simple(section, section_index)

def generate_hypothetical_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a hypothetical scenario question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a hypothetical question
    question = f"What if a Highly Automated Vehicle encounters a situation not explicitly covered in the {title.lower()} section of the Pennsylvania guidelines? How might the principles in this section be applied to this novel scenario?"
    
    # Generate an answer based on hypothetical scenario
    answer = f"If a Highly Automated Vehicle encounters a situation not explicitly covered in the {title.lower()} section, the underlying principles would still apply. The vehicle would need to prioritize safety, follow the general intent of the guidelines, and implement a reasonable interpretation of the requirements. Specifically, it would need to: 1) Apply the closest analogous rule from the guidelines; 2) Default to the most conservative safety approach; 3) Document the situation for future regulatory consideration; and 4) Ensure that its response aligns with the overall regulatory framework's goals of ensuring public safety while enabling technological advancement."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the core principles and intent behind the {title.lower()} requirements in the guidelines.",
        f"Next, I need to extrapolate how these principles would logically extend to situations not explicitly covered.",
        f"Finally, I can formulate a plausible response based on the document's overall regulatory philosophy and safety priorities."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "HYPOTHETICAL",
        "difficulty": "HARD",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_summarization_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a summarization question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Generate a summarization question
    question = f"Provide a concise summary (under 100 words) of the key points regarding {title.lower()} in the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles."
    
    # Generate a summary answer
    answer = f"The Pennsylvania guidelines on {title.lower()} for Highly Automated Vehicles can be summarized as follows: The section establishes key requirements for {title.lower()}, emphasizing safety protocols, compliance standards, and procedural requirements. It outlines responsibilities for operators, testing parameters, and documentation needs. The guidelines prioritize public safety while enabling technological advancement through a structured regulatory approach that balances innovation with risk management."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the most important information about {title.lower()} from the guidelines.",
        f"Next, I need to condense this information into its essential components, eliminating redundancy and minor details.",
        f"Finally, I can craft a concise summary that captures the key points in under 100 words."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "SUMMARIZATION",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_critical_analysis_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a critical analysis question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a critical analysis question that prompts evaluation of the document content. The question should test the reader's ability to critically assess the strengths and weaknesses of the material.

Follow these steps:
1. Analyze the content to identify potential strengths, weaknesses, assumptions, or limitations.
2. Create a question that asks for a critical evaluation of specific aspects of the content.
3. Provide an answer that thoughtfully assesses the strengths and weaknesses of the identified aspects.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "CRITICAL_ANALYSIS",
  "difficulty": "HARD",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating critical analysis QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_critical_analysis_qa_simple(section, section_index)

def generate_critical_analysis_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a critical analysis question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a critical analysis question
    question = f"What are the potential limitations or weaknesses in the {title.lower()} section of the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles? How might these limitations impact implementation?"
    
    # Generate an answer based on critical analysis
    answer = f"The {title.lower()} section of the Pennsylvania guidelines has several potential limitations: 1) It may not fully address rapidly evolving technologies, creating regulatory gaps for new innovations; 2) The requirements could be overly prescriptive in some areas, potentially stifling innovation; 3) There may be insufficient consideration of edge cases or unusual scenarios; 4) The guidelines might create compliance burdens that disproportionately impact smaller companies; and 5) There could be ambiguities in terminology that lead to inconsistent interpretation. These limitations could impact implementation by creating uncertainty for manufacturers, potentially delaying deployment of beneficial technologies, or requiring frequent regulatory updates as the technology evolves."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to critically examine the {title.lower()} section to identify potential gaps, inconsistencies, or problematic assumptions.",
        f"Next, I need to assess how these limitations might affect different stakeholders and the overall effectiveness of the regulations.",
        f"Finally, I can evaluate the potential consequences of these limitations for the implementation and evolution of Highly Automated Vehicle technology."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "CRITICAL_ANALYSIS",
        "difficulty": "HARD",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_technical_explanation_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a technical explanation question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a question asking to explain a complex technical process in layman's terms. The question should test the reader's ability to simplify technical concepts for a non-specialist audience.

Follow these steps:
1. Analyze the content to identify complex technical processes, concepts, or terminology.
2. Create a question that asks for an explanation of this technical content in simple terms.
3. Provide an answer that explains the concept clearly without jargon, as if teaching it to a high school student.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "TECHNICAL_EXPLANATION",
  "difficulty": "MEDIUM",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating technical explanation QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_technical_explanation_qa_simple(section, section_index)

def generate_technical_explanation_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a technical explanation question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a technical explanation question
    question = f"Explain the technical aspects of {title.lower()} for Highly Automated Vehicles in simple terms, as if you were teaching a high school student with no prior knowledge of autonomous vehicle technology."
    
    # Generate an answer based on technical explanation
    answer = f"Think of {title.lower()} for self-driving cars like this: Imagine you're teaching someone to drive. First, you'd explain the rules of the road, show them how the car works, and then watch them practice until they're safe. For self-driving cars, {title.lower()} works similarly. The car's computer system needs to learn all the rules, demonstrate it can follow them correctly, and prove it can handle unexpected situations safely. The Pennsylvania guidelines create a structured way to ensure these computer drivers are properly trained and tested before they're allowed on public roads. They specify what the car needs to know, how it should behave in different situations, and what safety measures must be in place to protect everyone on the road."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the technical concepts related to {title.lower()} in the guidelines.",
        f"Next, I need to translate these technical concepts into everyday language and relatable analogies.",
        f"Finally, I can structure an explanation that builds understanding progressively without using specialized terminology."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "TECHNICAL_EXPLANATION",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_process_workflow_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a process/workflow question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate a question about sequential processes or workflows. The question should test the reader's ability to understand and recall the steps in a process in the correct order.

Follow these steps:
1. Analyze the content to identify any sequential processes, procedures, or workflows.
2. Create a question that asks about the stages or steps in this process.
3. Provide an answer that lists the steps in the correct order with phase names.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "PROCESS_WORKFLOW",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating process workflow QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_process_workflow_qa_simple(section, section_index)

def generate_true_false_fill_blank_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a true/false or fill-in-the-blank question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate either a true/false statement with corrections for false statements, or a fill-in-the-blank question with key terms. The question should test the reader's basic recall and understanding of key facts.

Follow these steps:
1. Analyze the content to identify key facts, definitions, or statements.
2. Create either:
   a. A true/false statement based on the content (with a correction if false), or
   b. A fill-in-the-blank question where a key term is missing.
3. Provide the correct answer, including the full corrected statement for false statements.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "TRUE_FALSE_FILL_BLANK",
  "difficulty": "EASY",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating true/false or fill-in-the-blank QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_true_false_fill_blank_qa_simple(section, section_index)

def generate_contextual_ambiguity_qa_with_lmdeploy(pipe, section: str, section_index: int) -> Dict[str, Any]:
    """Generate a contextual ambiguity resolution question-answer pair using lmdeploy."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Extract the content without the title
    content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
    
    # Prepare the prompt
    prompt = f"""
For the following document section about Highly Automated Vehicles (HAV) regulations, generate an ambiguous question where the answer depends on understanding context from multiple parts of the section. The question should test the reader's ability to resolve ambiguities by considering the broader context.

Follow these steps:
1. Analyze the content to identify terms, phrases, or references that could be ambiguous without proper context.
2. Create a question that asks about the meaning or referent of such an ambiguous element.
3. Provide an answer that explains how the context resolves the ambiguity.
4. Use markdown format for the answer where appropriate.

Use the following JSON format:

{{
  "section_number": "{section_index}",
  "question_type": "CONTEXTUAL_AMBIGUITY",
  "difficulty": "HARD",
  "question": "<text>",
  "answer": "<text>",
  "reasoning_steps": [
    "<Step 1 description>",
    "<Step 2 description>",
    "<Step 3 description>"
  ]
}}

Here is the section content:

{title}

{content}
"""
    
    # Call lmdeploy
    try:
        from lmdeploy import GenerationConfig
        
        # Set up generation config
        gen_config = GenerationConfig(temperature=args.temperature, max_new_tokens=args.max_tokens)
        
        # Generate response
        response = pipe(prompt, gen_config=gen_config)
        
        # Extract the JSON from the response
        json_str = get_json_str(response.text)
        qa_pair = json.loads(json_str)
        
        return qa_pair
    
    except Exception as e:
        print(f"Error generating contextual ambiguity QA pair with lmdeploy: {e}")
        # Fallback to simple heuristic method
        return generate_contextual_ambiguity_qa_simple(section, section_index)

def generate_process_workflow_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a process/workflow question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a process/workflow question
    question = f"What are the sequential steps or stages in the {title.lower()} process for Highly Automated Vehicles according to the Pennsylvania Department of Transportation guidelines? List them in order."
    
    # Generate an answer based on process/workflow
    answer = f"The {title.lower()} process for Highly Automated Vehicles consists of the following sequential steps: 1) Initial Assessment - Evaluate the vehicle's capabilities and intended operational domain; 2) Documentation Preparation - Compile all required technical specifications and safety protocols; 3) Submission Phase - Submit materials to regulatory authorities for review; 4) Evaluation Period - Authorities assess compliance with all requirements; 5) Feedback Integration - Address any concerns or requested modifications; 6) Final Approval - Receive authorization to proceed with the next phase; and 7) Ongoing Monitoring - Maintain compliance through regular reporting and updates."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify any sequential processes or workflows described in the {title.lower()} section.",
        f"Next, I need to organize these steps in their correct chronological or logical order.",
        f"Finally, I can present these steps with clear phase names and descriptions to show the complete workflow."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "PROCESS_WORKFLOW",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_true_false_fill_blank_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a true/false or fill-in-the-blank question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Randomly choose between true/false or fill-in-the-blank
    is_true_false = random.choice([True, False])
    
    if is_true_false:
        # Generate a true/false question
        # Randomly decide if the statement should be true or false
        is_statement_true = random.choice([True, False])
        
        if is_statement_true:
            question = f"True or False: The Pennsylvania Department of Transportation guidelines require specific {title.lower()} protocols for Highly Automated Vehicles to ensure public safety."
            answer = f"True. The Pennsylvania Department of Transportation guidelines do require specific {title.lower()} protocols for Highly Automated Vehicles to ensure public safety."
        else:
            question = f"True or False: The Pennsylvania Department of Transportation guidelines exempt Highly Automated Vehicles from all {title.lower()} requirements."
            answer = f"False. Correction: The Pennsylvania Department of Transportation guidelines do NOT exempt Highly Automated Vehicles from {title.lower()} requirements. In fact, they establish specific standards and protocols that must be followed."
    else:
        # Generate a fill-in-the-blank question
        question = f"According to the Pennsylvania Department of Transportation guidelines, the primary purpose of {title.lower()} requirements for Highly Automated Vehicles is to ensure ____________."
        answer = f"public safety and regulatory compliance"
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify a key fact or concept related to {title.lower()} in the guidelines.",
        f"Next, I need to formulate this information into a clear statement that can be assessed as true/false or completed with a specific term.",
        f"Finally, I need to provide the correct answer with appropriate explanation or correction if needed."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "TRUE_FALSE_FILL_BLANK",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_contextual_ambiguity_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a contextual ambiguity resolution question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a contextual ambiguity question
    question = f"In the {title.lower()} section of the Pennsylvania Department of Transportation guidelines, what does 'the system' refer to in the context of Highly Automated Vehicles? Why might this reference be ambiguous without proper context?"
    
    # Generate an answer based on contextual ambiguity
    answer = f"In the {title.lower()} section, 'the system' refers specifically to the autonomous driving system that controls the vehicle's operations, not to the broader regulatory system or the vehicle's mechanical systems. This reference could be ambiguous because without proper context, 'the system' could potentially refer to: 1) The autonomous driving software; 2) The vehicle's hardware components; 3) The regulatory framework governing HAVs; 4) The testing and certification system; or 5) The emergency response system. The correct interpretation requires understanding the surrounding context, which clarifies that it refers to the autonomous driving system responsible for vehicle navigation and decision-making."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify potentially ambiguous terms or references in the {title.lower()} section.",
        f"Next, I need to analyze how the surrounding context helps resolve this ambiguity and determine the correct interpretation.",
        f"Finally, I can explain why this term could be confusing without proper context and how a careful reading of the full section clarifies its meaning."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "CONTEXTUAL_AMBIGUITY",
        "difficulty": "HARD",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_application_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate an application/practical use question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate an application question
    question = f"A company is developing a new Highly Automated Vehicle and needs to comply with the {title.lower()} requirements in Pennsylvania. What specific steps should they follow according to the guidelines?"
    
    # Generate an answer based on application
    answer = f"To comply with the {title.lower()} requirements in Pennsylvania, the company should follow these steps: 1) Review the specific criteria outlined in the guidelines; 2) Develop internal processes that ensure all aspects of {title.lower()} are addressed; 3) Document how their vehicle design meets each requirement; 4) Implement testing procedures to verify compliance; 5) Prepare all necessary documentation for submission to regulatory authorities; 6) Submit the required materials through the designated channels; 7) Address any feedback or requests for additional information; and 8) Maintain ongoing compliance through regular monitoring and updates."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the practical requirements related to {title.lower()} in the guidelines.",
        f"Next, I need to translate these regulatory requirements into actionable steps a company could follow.",
        f"Finally, I can organize these steps into a logical sequence that would ensure compliance with the {title.lower()} requirements."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "APPLICATION",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_comparative_analysis_qa_simple(sections: List[str], section_index: int) -> Dict[str, Any]:
    """Generate a comparative analysis question-answer pair using simple heuristics."""
    # Need at least two sections for comparative analysis
    if len(sections) < 2:
        return None
    
    # Select two random sections
    section1, section2 = random.sample(sections, 2)
    
    # Extract the section titles if they exist
    title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
    title1 = title_match1.group(1).strip() if title_match1 else "Section A"
    
    title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
    title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
    # Generate a comparative analysis question
    question = f"Compare and contrast the requirements for {title1.lower()} and {title2.lower()} as outlined in the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles. What are the key similarities and differences?"
    
    # Generate an answer based on comparative analysis
    answer = f"When comparing {title1.lower()} and {title2.lower()}, several key similarities and differences emerge. Similarities include: both require detailed documentation, both are subject to regulatory oversight, and both contribute to overall vehicle safety. Key differences include: {title1.lower()} focuses more on technical specifications while {title2.lower()} emphasizes procedural requirements; {title1.lower()} has more quantitative standards, whereas {title2.lower()} involves more qualitative assessments; and the verification processes for each have different timelines and responsible parties."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the key characteristics and requirements for both {title1.lower()} and {title2.lower()}.",
        f"Next, I need to systematically analyze which aspects are shared between them and which are distinct.",
        f"Finally, I can organize these observations into a structured comparison highlighting both similarities and differences."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "COMPARATIVE_ANALYSIS",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_cause_effect_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate a cause-effect relationship question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate a cause-effect question
    question = f"According to the Pennsylvania Department of Transportation guidelines, what consequences or effects result from non-compliance with the {title.lower()} requirements for Highly Automated Vehicles?"
    
    # Generate an answer based on cause-effect
    answer = f"Non-compliance with the {title.lower()} requirements leads to several consequences: 1) Potential denial or revocation of operating permits; 2) Mandatory cessation of vehicle testing or deployment; 3) Financial penalties as prescribed by regulatory authorities; 4) Increased scrutiny and more frequent inspections for future operations; and 5) Possible legal liability in the event of accidents or incidents related to the non-compliance."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the causal relationships described in the section on {title.lower()}.",
        f"Next, I need to trace the chain of effects that result from specific actions or conditions related to {title.lower()}.",
        f"Finally, I can articulate the complete cause-effect relationship, showing how one event or condition leads to specific outcomes."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "CAUSE_EFFECT",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_inference_qa_simple(section: str, section_index: int) -> Dict[str, Any]:
    """Generate an inference question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Generate an inference question
    question = f"Based on the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles, what can be inferred about the importance of {title.lower()} in ensuring public safety?"
    
    # Generate an answer based on inference
    answer = f"While not explicitly stated, we can infer from the guidelines that {title.lower()} plays a critical role in ensuring public safety because the detailed requirements and procedures outlined suggest that proper implementation is essential for preventing accidents and protecting road users. The emphasis placed on documentation, verification, and compliance indicates that regulators consider {title.lower()} to be a fundamental component of the safety framework for autonomous vehicles. Furthermore, the specific standards established imply that inadequate attention to {title.lower()} could potentially compromise the vehicle's ability to operate safely in various conditions and scenarios."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to analyze what the guidelines imply about {title.lower()} beyond what is explicitly stated.",
        f"By examining the level of detail and emphasis placed on {title.lower()}, I can draw logical conclusions about its importance.",
        f"I can then articulate these inferences to show the implicit significance of {title.lower()} in the context of public safety."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "INFERENCE",
        "difficulty": "MEDIUM",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_factual_recall_qa_simple(facts: List[str], section: str, section_index: int) -> Dict[str, Any]:
    """Generate a factual recall question-answer pair using simple heuristics."""
    # Extract the section title if it exists
    title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "Section"
    
    # Select a random fact from the section
    fact = random.choice(facts) if facts else ""
    
    # Generate a factual recall question
    question = f"According to the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles, what specific information is provided about {title.lower()}?"
    
    # Generate an answer based on the fact
    answer = f"The guidelines explicitly state: \"{fact}\""
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to identify the specific factual information provided in the section about {title.lower()}.",
        f"The document contains explicit statements about {title.lower()} that can be directly quoted.",
        f"I can provide the exact text from the document that answers this factual question."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "FACTUAL_RECALL",
        "difficulty": "EASY",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def generate_multi_hop_qa_simple(sections: List[str], section_index: int) -> Dict[str, Any]:
    """Generate a multi-hop reasoning question-answer pair using simple heuristics."""
    # Need at least two sections for multi-hop reasoning
    if len(sections) < 2:
        return None
    
    # Select two random sections
    section1, section2 = random.sample(sections, 2)
    
    # Extract the section titles if they exist
    title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
    title1 = title_match1.group(1).strip() if title_match1 else "Section A"
    
    title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
    title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
    # Generate a multi-hop reasoning question
    question = f"How do the requirements for {title1.lower()} influence or relate to the procedures outlined for {title2.lower()} in the Pennsylvania Department of Transportation guidelines for Highly Automated Vehicles?"
    
    # Generate an answer based on multi-hop reasoning
    answer = f"The requirements for {title1.lower()} directly influence the procedures for {title2.lower()} in several ways. First, the guidelines establish that {title1.lower()} sets foundational standards that must be met before {title2.lower()} can be properly implemented. Additionally, the documentation requirements for {title1.lower()} provide essential information needed for the verification processes described in the {title2.lower()} section. Furthermore, compliance with {title1.lower()} standards is a prerequisite for advancing through the stages outlined in the {title2.lower()} procedures. This interconnection demonstrates how the regulatory framework creates a cohesive approach to ensuring that Highly Automated Vehicles meet all necessary safety and operational requirements."
    
    # Generate reasoning steps
    reasoning_steps = [
        f"First, I need to understand the key points from both the {title1.lower()} and {title2.lower()} sections of the guidelines.",
        f"Next, I need to identify the connections and relationships between these two different aspects of Highly Automated Vehicles regulation.",
        f"Finally, I can explain how information from both sections must be combined to fully understand the regulatory framework."
    ]
    
    return {
        "section_number": str(section_index),
        "question_type": "MULTI_HOP_REASONING",
        "difficulty": "HARD",
        "question": question,
        "answer": answer,
        "reasoning_steps": reasoning_steps
    }

def output_to_phi_format(data: List[Dict[str, Any]]) -> Tuple[List[List[Dict[str, str]]], List[List[Dict[str, str]]], List[List[Dict[str, str]]], List[List[Dict[str, str]]]]:
    """
    Converts JSON data to different formats for different levels of Chain of Thought (CoT) reasoning steps.
    """
    no_cot, cot_one_step, cot_two_steps, cot_all_steps = [], [], [], []

    for item in data:
        question = item['question']
        answer = item['answer']
        reasoning_steps = item['reasoning_steps']

        response_no_cot = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

        response_cot_one_step = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f'{reasoning_steps[0]}\n{answer}'}
        ]

        response_cot_two_steps = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f'{reasoning_steps[0]} {reasoning_steps[1]}\n{answer}'}
        ]

        response_cot_all_steps = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f'{reasoning_steps[0]} {reasoning_steps[1]} {reasoning_steps[2]}\n{answer}'}
        ]

        no_cot.append(response_no_cot)
        cot_one_step.append(response_cot_one_step)
        cot_two_steps.append(response_cot_two_steps)
        cot_all_steps.append(response_cot_all_steps)

    return no_cot, cot_one_step, cot_two_steps, cot_all_steps

def generate_question_for_section(pipe, question_type: QuestionType, section: str, sections: List[str], section_index: int, facts: List[str]) -> Optional[Dict[str, Any]]:
    """Generate a question of the specified type for the given section."""
    try:
        if question_type == QuestionType.FACTUAL_RECALL:
            if args.no_llm:
                return generate_factual_recall_qa_simple(facts, section, section_index)
            else:
                return generate_factual_recall_qa_with_lmdeploy(pipe, facts, section, section_index)
        elif question_type == QuestionType.INFERENCE:
            if args.no_llm:
                return generate_inference_qa_simple(section, section_index)
            else:
                return generate_inference_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.MULTI_HOP_REASONING:
            if args.no_llm:
                return generate_multi_hop_qa_simple(sections, section_index)
            else:
                return generate_multi_hop_qa_with_lmdeploy(pipe, sections, section_index)
        elif question_type == QuestionType.APPLICATION:
            if args.no_llm:
                return generate_application_qa_simple(section, section_index)
            else:
                return generate_application_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.COMPARATIVE_ANALYSIS:
            if args.no_llm:
                return generate_comparative_analysis_qa_simple(sections, section_index)
            else:
                return generate_comparative_analysis_qa_with_lmdeploy(pipe, sections, section_index)
        elif question_type == QuestionType.CAUSE_EFFECT:
            if args.no_llm:
                return generate_cause_effect_qa_simple(section, section_index)
            else:
                return generate_cause_effect_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.SUMMARIZATION:
            if args.no_llm:
                return generate_summarization_qa_simple(section, section_index)
            else:
                return generate_summarization_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.HYPOTHETICAL:
            if args.no_llm:
                return generate_hypothetical_qa_simple(section, section_index)
            else:
                return generate_hypothetical_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.CRITICAL_ANALYSIS:
            if args.no_llm:
                return generate_critical_analysis_qa_simple(section, section_index)
            else:
                return generate_critical_analysis_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.TECHNICAL_EXPLANATION:
            if args.no_llm:
                return generate_technical_explanation_qa_simple(section, section_index)
            else:
                return generate_technical_explanation_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.PROCESS_WORKFLOW:
            if args.no_llm:
                return generate_process_workflow_qa_simple(section, section_index)
            else:
                return generate_process_workflow_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.TRUE_FALSE_FILL_BLANK:
            if args.no_llm:
                return generate_true_false_fill_blank_qa_simple(section, section_index)
            else:
                return generate_true_false_fill_blank_qa_with_lmdeploy(pipe, section, section_index)
        elif question_type == QuestionType.CONTEXTUAL_AMBIGUITY:
            if args.no_llm:
                return generate_contextual_ambiguity_qa_simple(section, section_index)
            else:
                return generate_contextual_ambiguity_qa_with_lmdeploy(pipe, section, section_index)
        else:
            print(f"Unsupported question type: {question_type}")
            return None
    except Exception as e:
        print(f"Error generating question of type {question_type}: {e}")
        return None

def main():
    # Set up lmdeploy pipeline if using LLM
    pipe = None
    if not args.no_llm:
        pipe = setup_lmdeploy_pipeline()
        if pipe is None:
            print("Failed to set up lmdeploy pipeline. Falling back to simple heuristics.")
            args.no_llm = True
    
    # Lists to store QA pairs
    fact_qa_pairs = []  # Original fact-based QA pairs
    summary_qa_pairs = []  # Original summary QA pairs
    new_qa_pairs = []  # New question types
    
    # Process each markdown file in the directory
    for filename in os.listdir(MARKDOWN_DOCS_DIR):
        if filename.endswith('.md'):
            file_path = os.path.join(MARKDOWN_DOCS_DIR, filename)
            print(f"Processing {file_path}...")
            
            # Read the markdown content
            content = read_markdown_file(file_path)
            
            # Split into sections
            sections = split_into_sections(content)
            
            # Process each section
            for section_idx, section in enumerate(sections):
                section_idx += 1  # 1-based indexing for section numbers
                print(f"  Processing section {section_idx}/{len(sections)}...")
                
                # Extract facts from the section
                facts = extract_facts_from_section(section)
                print(f"    Found {len(facts)} facts in section {section_idx}")
                
                # Generate original fact-based QA pairs
                for fact_idx, fact in enumerate(facts):
                    fact_idx += 1  # 1-based indexing for fact numbers
                    print(f"    Generating fact-based QA pair for fact {fact_idx}/{len(facts)} in section {section_idx}...")
                    
                    if args.no_llm:
                        qa_pair = generate_fact_qa_pair_simple(fact, section_idx, fact_idx)
                    else:
                        qa_pair = generate_fact_qa_pair_with_lmdeploy(pipe, fact, section_idx, fact_idx)
                        # Add a small delay to avoid rate limiting
                        time.sleep(1)
                    
                    fact_qa_pairs.append(qa_pair)
                
                # Generate original summary QA pair
                print(f"    Generating summary QA pair for section {section_idx}...")
                if args.no_llm:
                    summary_qa_pair = generate_section_summary_qa_pair_simple(section, section_idx)
                else:
                    summary_qa_pair = generate_section_summary_qa_pair_with_lmdeploy(pipe, section, section_idx)
                    # Add a small delay to avoid rate limiting
                    time.sleep(1)
                
                summary_qa_pairs.append(summary_qa_pair)
                
                # Generate new question types
                # Determine how many questions of each type to generate for this section
                # total_new_questions = min(len(facts), 10)  # Cap at 10 new questions per section
                total_new_questions = len(facts) # Use the number of facts as the number of new questions to generate
                
                # Calculate the number of questions for each type based on distribution
                question_counts = {}
                for q_type, percentage in QUESTION_TYPE_DISTRIBUTION.items():
                    count = max(1, int(total_new_questions * percentage / 100))
                    question_counts[q_type] = count
                
                # Generate questions for each type
                for q_type, count in question_counts.items():
                    for i in range(count):
                        print(f"    Generating {q_type.name} question {i+1}/{count} for section {section_idx}...")
                        qa_pair = generate_question_for_section(pipe, q_type, section, sections, section_idx, facts)
                        if qa_pair:
                            new_qa_pairs.append(qa_pair)
                            # Add a small delay to avoid rate limiting
                            if not args.no_llm:
                                time.sleep(1)
    
    print(f"Generated {len(fact_qa_pairs)} fact QA pairs")
    print(f"Generated {len(summary_qa_pairs)} summary QA pairs")
    print(f"Generated {len(new_qa_pairs)} new question type QA pairs")
    
    # Combine all QA pairs
    all_qa_pairs = fact_qa_pairs + new_qa_pairs
    
    # Convert to different CoT formats
    print("Converting all QA pairs to different CoT formats for training...")
    all_responses = output_to_phi_format(all_qa_pairs)
    
    print("Converting summary QA pairs to different CoT formats for validation...")
    summary_responses = output_to_phi_format(summary_qa_pairs)
    
    # Prepare training and validation data
    train_data = []
    eval_data = []
    
    # Add all QA pairs to training data
    for i in range(4):  # For each CoT format
        train_data.extend(all_responses[i])
    
    # Add all summary QA pairs to validation data
    for i in range(4):  # For each CoT format
        eval_data.extend(summary_responses[i])
    
    # Get current timestamp for filenames
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save the training data
    train_file = os.path.join(OUTPUT_DIR, f'train-synthetic-data-{timestamp}.json')
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)
    
    # Save the evaluation data
    eval_file = os.path.join(OUTPUT_DIR, f'validation-synthetic-data-{timestamp}.json')
    with open(eval_file, "w") as f:
        json.dump(eval_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training examples to {train_file}")
    print(f"Saved {len(eval_data)} validation examples to {eval_file}")

if __name__ == "__main__":
    main()
