import os
import json
from typing import List, Dict, Any, Union, Optional
import nest_asyncio
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from utilities.logger import setup_logger
import re 

logger = setup_logger(__name__)

def setup_lmdeploy_pipeline(model: str, num_gpus: int = 1, model_format: Optional[str] = None):
    """Set up and return the lmdeploy pipeline."""
    try:
        
        # Apply nest_asyncio to allow nested event loops
        nest_asyncio.apply()
        
        # System prompt for the model
        system_prompt = f"""You are a meticulous synthetic data generator that creates high-quality training examples from document sections. Your role is to transform content into question-answer pairs while strictly adhering to these guidelines:

1. DOCUMENT AND EXHIBIT REFERENCES:
   - Never refer to documents, exhibits, figures, or tables directly in the questions.
   - In answers, reference sources minimally (maximum once per answer if absolutely necessary).
   - When referencing exhibits/figures, use exact format: "Exhibit X in [document title]" (e.g., "Exhibit 5.2 in Traffic Signal Timing Manual").
   - Always refer to documents in third person (e.g., "According to Traffic Signal Timing Manual" never "According to the document").
   - Never use first-person references ("I", "we", "our document").

2. IMAGE AND FILE HANDLING:
   - Only include images in responses when they are actual ATSPM charts that need interpretation.
   - When including images, you MUST use the exact file paths provided in the input without modification.
   - Never invent or modify file names - use them exactly as given (e.g., if input shows "figures/chart1.png", output must use same path).
   - For image interpretation questions, ensure the question clearly states what analysis is needed.

3. CONTENT GENERATION PRINCIPLES:
   - Maintain objective, third-person perspective throughout.
   - Questions should be self-contained and not require document access.
   - Answers should stand alone but may minimally reference sources when essential.
   - Preserve the original document's technical style and tone.
   - For technical processes, provide clear, step-by-step explanations.

4. OUTPUT FORMATTING:
   - Strictly adhere to the specified JSON format.
   - Crucially, all JSON keys and string values MUST be enclosed in double quotes (`"`). Single quotes (`'`) are not permitted and will result in a parsing error.
   - Use markdown formatting in answers for clarity (bullet points, numbered lists, etc.).
   - For process/workflow questions, present steps in logical sequence.
   - For comparative questions, structure answers to clearly show similarities/differences.

5. SPECIAL CASES:
   - For multi-hop reasoning: Clearly indicate when combining information from different sections.
   - For hypotheticals: Base scenarios on realistic extensions of document concepts.
   - For critical analysis: Provide balanced assessments of strengths/weaknesses.
   - For contextual ambiguity: Explicitly show how context resolves the ambiguity.

Remember: You are creating training data that must work independently of the source document. Each question-answer pair should be fully understandable without referring back to the original material, except where minimal source attribution is absolutely necessary.
"""
        
        # Initialize chat template configuration
        chat_template_config = ChatTemplateConfig('internvl2_5')
        chat_template_config.meta_instruction = system_prompt
        
        # Initialize backend configuration
        backend_config = TurbomindEngineConfig(tp=num_gpus, model_format=model_format)
        
        # Create the pipeline with the provided configurations
        pipe = pipeline(model, chat_template_config=chat_template_config, backend_config=backend_config)
        
        return pipe
    except ImportError as e:
        logger.error(f"Error importing lmdeploy: {e}")
        logger.error("Please install lmdeploy with: pip install lmdeploy[all]")
        return None
    except Exception as e:
        logger.error(f"Error setting up lmdeploy pipeline: {e}")
        return None

def call_lmdeploy_api(
    pipe: pipeline,
    prompts: List[Union[str, Dict[str, Any]]],
    max_tokens: int = 4096,
    top_k: int = 40
) -> List[str]:
    """Call the LMDeploy API with prompts (optionally including images)."""
    responses = []

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Running prompt {i} of {len(prompts)}")
        try:
            logger.debug("Creating message...")

            # Build the message content
            content = []
            
            # Add text prompt
            text_prompt = prompt["text"] if isinstance(prompt, dict) else prompt
            content.append({"type": "text", "text": text_prompt})
            
            # Add images if available
            if isinstance(prompt, dict) and "images" in prompt and prompt["images"]:
                for img_path in prompt["images"]:
                    logger.debug(f"Attaching image to prompt: {img_path}")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "max_dynamic_patch": 12,
                            "url": img_path
                        }
                    })
            
            messages = [{"role": "user", "content": content}]

            logger.debug("Sending to LM Deploy...")
            response = pipe(
                messages, 
                gen_config=GenerationConfig(
                    max_new_tokens=max_tokens,
                    top_k=top_k
                )
            )
            responses.append(response)
        
        except Exception as e:
            logger.error(f"Error calling LM Deploy API: {e}")
            raise
    
    return responses

def get_json_str(string: str) -> str:
    """Extract JSON string from text."""
    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return string[first:last + 1]

def sanitize_json_str(text: str) -> str:
    """Sanitizes JSON string by removing control characters before parsing"""

    # Remove control characters except \t, \n, \r
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)