"""

Synthetic Training Data Generator for LLM Fine-tuning

This script processes markdown files and generates synthetic question-answer pairs,
formatted for training Large Language Models (LLMs). It uses lmdeploy with InternVL3-8B model (by default) 
to generate QA pairs with different difficulty levels.

"""

import sys
import os
import json
import random
import re
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from utilities.logger import setup_logger
from utilities.api_utils import *
from utilities.file_utils import * 
from models.question_types import QuestionType, QUESTION_TYPE_DIFFICULTY
import argparse 
from utilities.check_image_urls import check_image_urls_in_file, check_all_output_files

# Load environment variables
load_dotenv()

# Set up env variables
MARKDOWN_DOCS_DIR = os.environ.get("MARKDOWN_DOCS_DIR", "markdown_docs")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "output")
MODEL = os.environ.get("MODEL", "OpenGVLab/InternVL3-8B")
MODEL_FORMAT = os.environ.get("MODEL_FORMAT", None)
NUM_GPUS = int(os.environ.get("NUM_GPUS", 1))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 50))

# Initialize logging 
logger = setup_logger(__name__)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize the failed sections
failed_sections = []

def extract_image_references(section_content: str) -> List[str]:
    """Finds all image references in markdown content using {{FIGURE_X}} syntax.
    
    Args:
        section_content: Markdown text to scan
    
    Returns:
        List of found reference names (without braces)
    
    Example:
        >>> extract_image_references("See {{FIGURE_1.2}} and {{TABLE_3}}")
        ['FIGURE_1.2', 'TABLE_3']
    """

    pattern = r'\{\{([^}]+)\}\}'
    return re.findall(pattern, section_content)

def resolve_image_paths(image_refs: List[str], document_name: str) -> List[str]:
    """Converts image references to valid filesystem paths.
    
    Args:
        image_refs: List of reference names from extract_image_references()
        document_name: Base document name for path resolution
    
    Returns:
        List of absolute paths to existing image files
    
    Notes:
        - Checks multiple possible locations including document subdirectories
        - Logs warnings for missing images
    """

    image_paths = []
    for ref in image_refs:
        # Try different possible paths
        possible_paths = [
            os.path.join(document_name, f"{ref}.png"),
            os.path.join(document_name, f"{ref}.jpg"),
            os.path.join(document_name, f"{ref}.jpeg"),
            os.path.join(MARKDOWN_DOCS_DIR, document_name, f"{ref}.png"),
            os.path.join(MARKDOWN_DOCS_DIR, document_name, f"{ref}.jpg"),
            os.path.join(MARKDOWN_DOCS_DIR, document_name, f"{ref}.jpeg"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_paths.append(path)
                logger.debug(f"Found image: {path}")

                break
        else:
            logger.warning(f"Image not found for reference {ref} in document {document_name}")
    
    return image_paths

def attach_images_to_section(section: str, document_name: str) -> Tuple[str, List[str]]:
    """Processes a markdown section to extract and resolve image references.
    
    Args:
        section: Markdown content containing image references
        document_name: Base document name for path resolution
    
    Returns:
        Tuple containing:
        - Original section content (unchanged)
        - List of resolved image paths (absolute paths)
    
    See Also:
        :func:`extract_image_references` for reference extraction
        :func:`resolve_image_paths` for path resolution logic
    """

    image_refs = extract_image_references(section)
    image_paths = resolve_image_paths(image_refs, document_name)
    
    return section, image_paths

def find_closest_sections(section: str, sections: List[str], section_idx: int) -> List[Dict[str, Any]]:
    """Identifies semantically similar sections using TF-IDF and cosine similarity.
    
    Args:
        section: Target section content to find neighbors for
        sections: All available sections from the document
        section_idx: 1-based index of the target section
    
    Returns:
        List of similar section contents (excluding the target section)
    
    Notes:
        - Uses sklearn's TF-IDF vectorizer with English stop words
        - Limits text length to first 2000 characters per section
        - Returns max 3 nearest neighbors
    """

    # Preprocess sections for KNN
    section_texts = [sec[:2000] for sec in sections]  # Limit text length
    current_section_idx = section_idx - 1  # Convert to 0-based index
    
    # Initialize TF-IDF and KNN
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_matrix = vectorizer.fit_transform(section_texts)
    
    # Find 3 most similar sections (excluding current section)
    knn = NearestNeighbors(n_neighbors=min(3, len(section_texts)-1), metric='cosine')
    knn.fit(tfidf_matrix)
    
    # Get nearest neighbors (excluding self)
    fact_vec = vectorizer.transform([section])
    distances, indices = knn.kneighbors(fact_vec)
    
    relevant_sections = []
    for idx in indices[0]:
        if idx != current_section_idx:  # Skip current section
            relevant_sections.append(section_texts[idx])

    return relevant_sections

def get_closest_sections_text(section: str, sections: List[str], section_idx: int, document_name: str = "") -> Tuple[str, List[str]]:
    """Retrieves text and images from related sections.
    
    Args:
        section: Target section content
        sections: All document sections
        section_idx: 1-based index of target section
        document_name: Optional document name for image resolution
    
    Returns:
        Tuple containing:
        - Concatenated text of related sections
        - List of all image paths from related sections
    
    Notes:
        Processes images in each related section through attach_images_to_section()
    """

    relevant_sections = find_closest_sections(section, sections, section_idx)
    
    processed_sections = []
    all_image_paths = []
    
    for sec in relevant_sections:
        # Process images in each section
        sec_content, image_paths = attach_images_to_section(sec, document_name)
        processed_sections.append(sec_content)
        all_image_paths.extend(image_paths)
    
    relevant_sections_text = '\n\n'.join(processed_sections) if processed_sections else 'No additional relevant sections found'
        
    return relevant_sections_text, all_image_paths

def generate_qa_pairs_bulk(pipe, sections: List[str], section_indices: List[int], question_type: QuestionType, all_sections: List[str], doc_metadata: Dict[str, Any], document_name: str = "") -> List[Dict[str, Any]]:
    """Generates QA pairs for multiple sections in a batch.
    
    Args:
        pipe: Initialized LMDeploy pipeline
        sections: List of markdown sections to process
        section_indices: Corresponding 1-based indices for sections
        question_type: Type of questions to generate (from QuestionType enum)
        all_sections: Complete document content for context
        doc_metadata: Document metadata dictionary
        document_name: Optional document identifier for image resolution
    
    Returns:
        List of QA pairs with structure:
        {
            "question": str,
            "answer": str,
            "question_images": List[str],
            "question_type": str,
            "difficulty": str,
            "section_idx": int
        }
    
    Notes:
        - Handles retries for failed sections automatically
        - For FACT_BASED questions, processes each fact individually
        - Multi-hop questions require ≥2 valid sections
    """

    prompts = []
    metadata = []
    
    # Get document title from metadata or use filename
    doc_title = doc_metadata.get('title', 'the document')
    
    # Add metadata to prompt context
    metadata_context = ""
    if doc_metadata:
        metadata_context = "\nDOCUMENT METADATA:\n"
        for k, v in doc_metadata.items():
            metadata_context += f"- {k}: {v}\n"
        metadata_context += "\n"

    # Prepare prompts for each section
    for section, section_idx in zip(sections, section_indices):

        logger.debug(f"Creating prompt for section {section_idx}...")
        
        # Process images in the section
        section_content, section_image_paths = attach_images_to_section(section, document_name)
        
        # Get related sections and their images
        relevant_sections_text, related_image_paths = get_closest_sections_text(
            section_content, all_sections, section_idx, document_name
        )

        # Combine all image paths
        all_image_paths = section_image_paths + related_image_paths
        
        # Create image context for prompts
        image_context = ""
        if all_image_paths:
            image_context = f"\nIMAGES ATTACHED TO THIS SECTION:\n"
            for i, path in enumerate(all_image_paths):
                image_name = os.path.splitext(os.path.basename(path))[0]  # Get name without extension
                image_context += f"Image {image_name}: {path}\n"
            image_context += "Note: The attached images provide visual context for the questions and answers. Please reference them appropriately in your responses.\n\n"

        facts = extract_facts_from_section(section_content)


        # Extract the section title if it exists
        title_match = re.match(r'^#+ (.*?)$', section, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else "Section"
        
        # Extract the content without the title
        content = re.sub(r'^#+ .*?$', '', section, count=1, flags=re.MULTILINE).strip()
        
        # Modify all prompt templates to include this instruction:
        common_instructions = f"""
ADDITIONAL INSTRUCTIONS:
1. Always refer to the document in third person as "{doc_title}" (e.g., "According to {doc_title}" instead of "According to the document")
2. Never use first-person references like "I", "we", or "our document" - maintain objective third-person perspective
3. For visual references:
   - Only include images if they are ATSPM charts requiring interpretation
   - Ensure image paths reference real files in the system
   - When mentioning exhibits, specify the document name (e.g., "As shown in {doc_title}'s Figure 3...")
4. Answer requirements:
   - Must be complete and self-contained
   - Should not require external context beyond what's provided
   - Use markdown formatting where appropriate (lists, bold, etc.)
5. Question requirements:
   - Should be clear and unambiguous
   - Must be answerable using only the provided content
   - Should test specific knowledge or reasoning skills
6. Style guidelines:
   - Match the document's technical level and tone
   - Avoid colloquial language
   - Use consistent terminology from the source material
7. For multi-part answers:
   - Structure logically with clear steps
   - Number sequential processes
   - Highlight key conclusions
"""
        
        # Create prompt based on question type
        if question_type == QuestionType.FACT_BASED:

            logger.debug(f"FACTS: {facts}")

            for fact_idx, fact in enumerate(facts):

                logger.debug(f"Creating prompt for fact {fact_idx} for section {section_idx}...")
                logger.debug(f"FACT: {fact}")

                relevant_sections_text_for_fact = get_closest_sections_text(fact, all_sections, section_idx, document_name=document_name)
            
                prompt = f"""
For the following fact from a document, generate a self-contained question/answer pair using the provided context.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text_for_fact}

FACT TO BASE QUESTION ON:
{fact}

{image_context}

INSTRUCTIONS:
1. Analyze the fact's context and style, including language patterns and tone.
2. Create a question that specifically targets this fact while being self-contained.
3. Provide a detailed answer that reflects the document's style and uses context when needed.
4. Use markdown format for the answer where appropriate.
5. Only reference the context when absolutely necessary for clarity.
6. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use this JSON format:
{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}
"""
                
                # Create the prompt input structure
                if all_image_paths:
                    prompt_input = {
                        "text": prompt,  # The generated prompt text
                        "images": all_image_paths  # List of image paths
                    }
                else:
                    # prompt_input = prompt  # Just text if no images
                    prompt_input = {
                        "text": prompt,  # The generated prompt text
                    }

                prompts.append(prompt_input)
                metadata.append({
                    "section_idx": section_idx,
                    "fact_number": fact_idx,
                    "question_type": question_type.name,
                    "difficulty": QUESTION_TYPE_DIFFICULTY[question_type].value
                })
        elif question_type == QuestionType.SECTION_SUMMARY:
            prompt = f"""
For the following section from a document, generate a summary question/answer pair. The question should ask for a comprehensive summary of the section, and the answer should summarize all key information in the section. Reflect the writing style, tone, and thematic elements of the original document without directly referencing or quoting the text. Follow these steps:

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

1. Analyze the section's style, including language patterns, tone, and structure.
2. Create a question that asks for a summary of the key information in this section.
3. Provide a detailed answer that summarizes all important information in the section.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Ensure all outputs are independent of the original text context. The question and answer should appear as standalone general knowledge content. Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section:

{title}

{content}
"""
        elif question_type == QuestionType.FACTUAL_RECALL:

            fact = random.choice(facts) if facts else content[:500]

            prompt = f"""
For the following document section, generate a factual recall question that can be answered by directly quoting 1-2 sentences from the document. The question should test the reader's ability to locate and recall explicit information from the text.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify key facts, definitions, or statements.
2. Create a question that targets specific information that can be directly quoted from the text.
3. Provide an answer that includes exact text excerpts.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}

And here is a specific fact from this section that you might focus on:

{fact}
"""
        elif question_type == QuestionType.INFERENCE:
            prompt = f"""
For the following document section, generate an inference question that requires logical reasoning from the text. The question should test the reader's ability to understand implicit information and draw conclusions that are not explicitly stated.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify underlying principles, implications, or connections.
2. Create a question that requires connecting ideas or inferring meaning beyond what is directly stated.
3. Provide an answer that explains the logical reasoning process without directly quoting the text.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.MULTI_HOP_REASONING:

            # Need at least two sections for multi-hop reasoning
            if len(all_sections) < 2:
                # TODO: SKIP THIS SECTION
                return None
            
            # Use current section and select one other random section
            section1 = section
            other_sections = [s for s in all_sections if s != section]
            section2 = random.choice(other_sections) if other_sections else None
            
            if not section2:
                return None
            
            # TODO: ATTACH IMAGES FOR ALL THE OHTER SECTIONS IN HERE
            
            # Extract the section titles if they exist
            title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
            title1 = title_match1.group(1).strip() if title_match1 else "Section A"
            
            title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
            title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
            # Extract the content without the titles
            content1 = re.sub(r'^#+ .*?$', '', section1, count=1, flags=re.MULTILINE).strip()
            content2 = re.sub(r'^#+ .*?$', '', section2, count=1, flags=re.MULTILINE).strip()
            
            prompt = f"""
For the following two document sections, generate a multi-hop reasoning question that requires combining information from both sections. The question should test the reader's ability to connect ideas across different parts of the document.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze both sections to identify related concepts or complementary information.
2. Create a question that requires integrating knowledge from both sections to answer correctly.
3. Provide an answer that explains how information from both sections contributes to the solution.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here are the sections:

Section 1: {title1}
{content1}

Section 2: {title2}
{content2}
"""
        elif question_type == QuestionType.APPLICATION:
            prompt = f"""
For the following document section, generate a scenario-based question asking how to apply a concept from the document. The question should test the reader's ability to apply theoretical knowledge to practical situations.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify concepts, procedures, or guidelines that could be applied in real-world scenarios.
2. Create a question that presents a realistic scenario where this knowledge would need to be applied.
3. Provide a step-by-step answer that explains how to apply the concept in the given scenario.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.COMPARATIVE_ANALYSIS:

            # Need at least two sections for multi-hop reasoning
            if len(all_sections) < 2:
                # TODO: SKIP THIS SECTION
                return None
            
            # Use current section and select one other random section
            section1 = section
            other_sections = [s for s in all_sections if s != section]
            section2 = random.choice(other_sections) if other_sections else None
            
            if not section2:
                return None
            
            # Extract the section titles if they exist
            title_match1 = re.match(r'^#+ (.*?)$', section1, re.MULTILINE)
            title1 = title_match1.group(1).strip() if title_match1 else "Section A"
            
            title_match2 = re.match(r'^#+ (.*?)$', section2, re.MULTILINE)
            title2 = title_match2.group(1).strip() if title_match2 else "Section B"
    
            # Extract the content without the titles
            content1 = re.sub(r'^#+ .*?$', '', section1, count=1, flags=re.MULTILINE).strip()
            content2 = re.sub(r'^#+ .*?$', '', section2, count=1, flags=re.MULTILINE).strip()

            prompt = f"""
For the following two document sections, generate a comparative analysis question that asks the reader to compare two concepts or approaches. The question should test the reader's ability to identify similarities and differences between related ideas.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze both sections to identify concepts, approaches, or requirements that can be meaningfully compared.
2. Create a question that asks for a comparison, highlighting specific aspects to focus on.
3. Provide an answer that systematically analyzes similarities and differences.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here are the sections:

Section 1: {title1}
{content1}

Section 2: {title2}
{content2}
"""
        elif question_type == QuestionType.CAUSE_EFFECT:
            prompt = f"""
For the following document section, generate a question about cause-effect chains described in the document. The question should test the reader's ability to understand causal relationships and their implications.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify cause-effect relationships, where one event or condition leads to specific outcomes.
2. Create a question that asks about these causal relationships, focusing on why certain effects occur.
3. Provide an answer that explains the causal chain, citing specific parts of the document.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.SUMMARIZATION:
            prompt = f"""
For the following document section, generate a summarization question that asks for a concise summary of multi-page content. The question should test the reader's ability to identify and synthesize key information.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify the main topics, key points, and essential information.
2. Create a question that asks for a concise summary of specific aspects of the content.
3. Provide an answer that summarizes the requested information in under 100 words.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.HYPOTHETICAL:
            prompt = f"""
For the following document section, generate a 'what-if' question that extends document concepts to new situations. The question should test the reader's ability to apply principles from the document to hypothetical scenarios.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify key principles, rules, or guidelines that could be applied to new situations.
2. Create a hypothetical scenario that extends beyond what is explicitly covered in the document.
3. Provide a plausible speculative answer based on the principles established in the document.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.CRITICAL_ANALYSIS:
            prompt = f"""
For the following document section, generate a critical analysis question that prompts evaluation of the document content. The question should test the reader's ability to critically assess the strengths and weaknesses of the material.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify potential strengths, weaknesses, assumptions, or limitations.
2. Create a question that asks for a critical evaluation of specific aspects of the content.
3. Provide an answer that thoughtfully assesses the strengths and weaknesses of the identified aspects.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.TECHNICAL_EXPLANATION:
            prompt = f"""
For the following document section, generate a question asking to explain a complex technical process in layman's terms. The question should test the reader's ability to simplify technical concepts for a non-specialist audience.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify complex technical processes, concepts, or terminology.
2. Create a question that asks for an explanation of this technical content in simple terms.
3. Provide an answer that explains the concept clearly without jargon, as if teaching it to a high school student.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.PROCESS_WORKFLOW:
            prompt = f"""
For the following document section, generate a question about sequential processes or workflows. The question should test the reader's ability to understand and recall the steps in a process in the correct order.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify any sequential processes, procedures, or workflows.
2. Create a question that asks about the stages or steps in this process.
3. Provide an answer that lists the steps in the correct order with phase names.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>", 
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.TRUE_FALSE_FILL_BLANK:
            prompt = f"""
For the following document section, generate either a true/false statement with corrections for false statements, or a fill-in-the-blank question with key terms. The question should test the reader's basic recall and understanding of key facts.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify key facts, definitions, or statements.
2. Create either:
   a. A true/false statement based on the content (with a correction if false), or
   b. A fill-in-the-blank question where a key term is missing.
3. Provide the correct answer, including the full corrected statement for false statements.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        elif question_type == QuestionType.CONTEXTUAL_AMBIGUITY:
            prompt = f"""
For the following document section, generate an ambiguous question where the answer depends on understanding context from multiple parts of the section. The question should test the reader's ability to resolve ambiguities by considering the broader context.

{metadata_context}
{common_instructions}
CONTEXT:
1. FULL SECTION WHERE FACT APPEARS:
{section}

2. RELEVANT SECTIONS FROM DOCUMENT:
{relevant_sections_text}

{image_context}

Follow these steps:
1. Analyze the content to identify terms, phrases, or references that could be ambiguous without proper context.
2. Create a question that asks about the meaning or referent of such an ambiguous element.
3. Provide an answer that explains how the context resolves the ambiguity.
4. Use markdown format for the answer where appropriate.
5. Only include images in the response JSON if they are Automated Traffic Signal Performance Measure charts that the question is asking to analyze.

Use the following JSON format:

{{
  "question": "<text>",
  "question_images": ['<folder/image1.jpg>', '<folder/image2.jpg>']
  "answer": "<text>"
}}

Here is the section content:

{title}

{content}
"""
        

        if question_type is not QuestionType.FACT_BASED:
            
            # Create the prompt input structure
            if all_image_paths:
                prompt_input = {
                    "text": prompt,  # The generated prompt text
                    "images": all_image_paths  # List of image paths
                }
            else:
                # prompt_input = prompt  # Just text if no images
                prompt_input = {
                    "text": prompt,  # The generated prompt text
                }

            prompts.append(prompt_input)
            if question_type is not QuestionType.SECTION_SUMMARY:
                metadata.append({
                    "section_idx": section_idx,
                    "question_type": question_type.name,
                    "difficulty": QUESTION_TYPE_DIFFICULTY[question_type].value
                })
            else:
                metadata.append({
                    "section_idx": section_idx,
                    "question_type": question_type.name,
                    "difficulty": QUESTION_TYPE_DIFFICULTY[question_type].value,
                    "is_summary": True
                })
        
    logger.info(f"Prompts to run: {len(prompts)}")

    # Call API for prompts 
    responses = call_lmdeploy_api(pipe, prompts)
    
    # Process responses
    qa_pairs = []
    current_section_facts = []  # Track all facts from current section
    current_section_meta = None  # Track metadata for current section
    current_section_failed = False  # Flag if current section has failures

    for response, meta in zip(responses, metadata):
        try:
            # Skip processing facts if we've already failed this section
            if current_section_failed and meta.get('question_type') == 'FACT_BASED' and meta['section_idx'] == current_section_meta['section_idx']:
                continue

            # For FACT_BASED questions, collect all facts from a section before adding
            if meta.get('question_type') == 'FACT_BASED':
                # If new section, reset tracking
                if current_section_meta is None or meta['section_idx'] != current_section_meta['section_idx']:
                    # If we have pending facts from previous section, add them
                    if current_section_facts:
                        qa_pairs.extend(current_section_facts)
                    current_section_facts = []
                    current_section_meta = meta
                    current_section_failed = False

                qa_pair = response
                json_str = get_json_str(response.text)
                qa_pair = json.loads(json_str)
                qa_pair.update(meta)
                current_section_facts.append(qa_pair)
                
                logger.info(f"Successfully parsed fact {meta['fact_number']} for section {meta['section_idx']}.")
            else:
                # Handle non-FACT_BASED questions normally
                qa_pair = response
                json_str = get_json_str(response.text)
                qa_pair = json.loads(json_str)
                qa_pair.update(meta)
                qa_pairs.append(qa_pair)
                
                logger.info(f"Successfully parsed response.")
        
        except Exception as e:
            logger.error(f"Error processing response for section {meta['section_idx']}: {e}")
            # For FACT_BASED, mark section as failed and discard collected facts
            if meta.get('question_type') == 'FACT_BASED':
                current_section_facts = []
                current_section_failed = True
                logger.warning(f"Section {meta['section_idx']} marked as failed - skipping remaining facts.")
            

            failed_sections.append({
                'section_idx': meta['section_idx'],
                'content': section,
                'question_type': meta.get('question_type', 'unknown'),
                'error': str(e)
            })

            logger.debug(response.text)
    
    # Add any remaining facts from the last section if they all succeeded
    if current_section_facts:
        qa_pairs.extend(current_section_facts)
    
    return qa_pairs

def retry_failed_sections(pipe, failed_sections, all_sections, metadata, document_name=""):
    """Handles retry logic for sections that failed initial processing.
    
    Args:
        pipe: Initialized LMDeploy pipeline
        failed_sections: List of failed sections with structure:
            {
                'section_idx': int,
                'content': str,
                'question_type': str,
                'error': str
            }
        all_sections: Complete document content
        metadata: Document metadata dictionary
        document_name: Optional document identifier
    
    Returns:
        List of successfully processed QA pairs from retries
    
    Notes:
        - Removes successfully processed sections from failed_sections list
        - Maintains original error tracking for persistent failures
    """

    logger.info("Retrying failed sections...")

    retry_results = []  # Store successful retries

    for section in failed_sections:
        try:
            logger.info(f"Retrying section {section['section_idx']} (question type: {section['question_type']})")

            # Remove the failed sections, they'll be readded in the generate_qa_pairs_bulk if they fail again
            failed_sections.remove(section)

            # Generate QA pairs for this retry
            qa_pairs = generate_qa_pairs_bulk(
                pipe,
                [section['content']],
                [section['section_idx']], 
                QuestionType[section['question_type']], 
                all_sections,
                metadata,
                document_name
            )

            if qa_pairs:
                logger.debug("Adding result to the retry results")
                retry_results.extend(qa_pairs)

        except Exception as e:
            logger.error(f"Failed again on section {section['section_idx']}: {e}")

    return retry_results  # Return the successfully processed QA pairs


def output_to_phi_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Converts QA pairs to OpenAI-compatible training format.
    
    Args:
        data: List of QA pairs from generate_qa_pairs_bulk()
    
    Returns:
        List of messages in format:
        {
            "messages": [
                {"role": "user", "content": [...]},
                {"role": "assistant", "content": "..."}
            ],
            "question_type": str,
            "difficulty": str
        }
    
    Notes:
        - Preserves image references in user content
        - Maintains original question metadata
    """

    formatted_data = []

    for item in data:
        question = item['question']
        question_images = item.get('question_images', [])
        answer = item['answer']
        question_type = item.get('question_type', 'UNKNOWN')
        difficulty = item.get('difficulty', 'UNKNOWN')

        # # Build reasoning content if steps exist
        # reasoning_content = ""
        # if reasoning_steps:

        #     # If the steps are in a list of dicts, convert to string
        #     step_strings = []
        #     for step in reasoning_steps:
        #         if isinstance(step, dict):
        #             # Convert dict to "key: value" pairs
        #             step_str = "\n".join(f"{k}: {v}" for k, v in step.items())
        #         else:
        #             step_str = str(step)
        #         step_strings.append(step_str)
            
        #     reasoning_content = f"<think>\n{'\\n\\n'.join(step_strings)}\n</think>\n"

        # Build user content with text and optional images
        user_content = [{"type": "text", "text": question}]
        if question_images:
            for img in question_images:
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img,
                        "detail": "auto"
                    }
                })

        response = {
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                },
                {
                    "role": "assistant",
                    "content": f"{answer}"
                }
            ],
            "question_type": question_type,
            "difficulty": difficulty
        }

        formatted_data.append(response)

    return formatted_data

def main():
    """Orchestrates the synthetic data generation pipeline.
    
    System Flow:
    1. Argument parsing (--check_urls mode)
    2. LMDeploy pipeline initialization
    3. Document processing loop:
       a. Image validation
       b. Section splitting
       c. Batched QA generation (all question types)
       d. Retry mechanism for failures
    4. Output generation (train/eval split)
    
    Environment Variables:
        MARKDOWN_DOCS_DIR: Input directory for markdown files
        OUTPUT_DIR: Output directory for generated QA pairs
        MODEL: LLM model identifier
        NUM_GPUS: GPU allocation count
        MAX_RETRIES: Maximum retry attempts per section
    
    Notes:
        - Processes each markdown file independently
        - Handles memory errors gracefully by skipping problematic files
        - Summary questions are automatically assigned to eval set
    """
    
    # Add argument parsing
    parser = argparse.ArgumentParser(description='Generate synthetic training data or check image URLs')
    parser.add_argument('--check_urls', action='store_true', 
                       help='Check image URLs in JSON files')
    parser.add_argument('--file', type=str, default='',
                       help='Specific file to check (optional)')
    args = parser.parse_args()

    if args.check_urls:
        if args.file:
            # Check single file
            check_image_urls_in_file(args.file)
        else:
            # Check all files in output directory
            check_all_output_files(OUTPUT_DIR)
        return
    
    # Set up lmdeploy pipeline if using LLM
    pipe = None

    pipe = setup_lmdeploy_pipeline(MODEL, NUM_GPUS, MODEL_FORMAT)
    if pipe is None:
        logger.error("Failed to set up lmdeploy pipeline - exiting.")
        sys.exit(1)
    
    # Check to make sure all the images in each markdown file are valid so we can check for errors before continuing
    for filename in os.listdir(MARKDOWN_DOCS_DIR):
        if filename.endswith('.md'):
            file_path = os.path.join(MARKDOWN_DOCS_DIR, filename)
            
            document_name = get_document_name_from_filename(filename)
            logger.info(f"Checking images in {document_name}...")
            
            # Read the markdown content
            metadata, content = read_markdown_file(file_path)
            image_refs = extract_image_references(content)
            image_paths = resolve_image_paths(image_refs, document_name)

    # Process each markdown file in the directory
    for filename in os.listdir(MARKDOWN_DOCS_DIR):
        try:
            # Lists to store QA pairs, reset for each document
            new_qa_pairs = []  # New question types
            
            if filename.endswith('.md'):
                file_path = os.path.join(MARKDOWN_DOCS_DIR, filename)
                logger.info(f"Processing {file_path}...")
                
                # Read the markdown content
                # TODO: INSTEAD OF LOADING CONTENT INTO MEMORY, GENERATE ALL THE PROMPTS AND STORE THEM, THEN REMOVE THEM AS THEY ARE COMPLETED.
                # THE ISSUE IS THAT RIGHT NOW IF THE PROCESS GETS INTERRUPTED, THE FILE PROCESSING NEEDS TO START OVER, WHICH COULD BE HOURS OF LOST WORK.
                metadata, content = read_markdown_file(file_path)
                
                # Extract document name for image folder check
                document_name = get_document_name_from_filename(filename)
                logger.debug(f"Document name for image check: {document_name}")
                
                # Split into sections
                sections = split_into_sections(content)

                # Process sections in bulk
                batch_size = 100  # Adjust based on your API rate limits
                
                for question_type in QuestionType:

                    logger.info(f"Generating {question_type.name} questions in bulk...")
                    
                    # Prepare batches
                    for i in range(0, len(sections), batch_size):
                        batch_sections = sections[i:i+batch_size]
                        batch_indices = list(range(i+1, i+len(batch_sections)+1))  # 1-based
                        
                        qa_pairs = generate_qa_pairs_bulk(
                            pipe,
                            batch_sections,
                            batch_indices,
                            question_type,
                            sections,
                            metadata,
                            document_name
                        )

                        # Call this at the end of your main processing
                        max_retries = MAX_RETRIES
                        retry_count = 0
                        while len(failed_sections) > 0 and retry_count < max_retries:
                            logger.info(f"\nRetry attempt {retry_count + 1}/{max_retries} for {len(failed_sections)} failed sections...")
                            retry_results = retry_failed_sections(pipe, failed_sections, sections, metadata, document_name)
                            qa_pairs.extend(retry_results)
                            retry_count += 1
                        
                        # Log any remaining failures after max retries
                        if len(failed_sections) > 0:
                            logger.warning(f"{len(failed_sections)} sections failed after {max_retries} retries")
                        
                        # Add to results
                        new_qa_pairs.extend(qa_pairs)
        
                # Convert QA pairs to JSON format
                logger.info("Converting all QA pairs to JSON format for training...")
                all_responses = output_to_phi_format(new_qa_pairs)
                
                # Prepare training and validation data
                train_data = []
                eval_data = []

                # Separate summary QA pairs for evaluation and others for training
                for qa_pair in all_responses:
                    if qa_pair['question_type'] == "SECTION_SUMMARY":
                        eval_data.append(qa_pair)
                    else:
                        train_data.append(qa_pair)
                
                # Save the output
                train_file = save_output_to_file(train_data, OUTPUT_DIR, f'train-{document_name}')
                eval_file = save_output_to_file(eval_data, OUTPUT_DIR, f'eval-{document_name}')
                
                logger.info(f"Saved {len(train_data)} training examples to {train_file}")
                logger.info(f"Saved {len(eval_data)} validation examples to {eval_file}")
            
        except MemoryError:
            logger.error(f"Memory error processing {filename}, skipping to next file")
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing {filename}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
