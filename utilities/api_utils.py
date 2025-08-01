import os
import json
from typing import List, Dict, Any, Union, Optional
import nest_asyncio
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig, GenerationConfig
from openai import OpenAI
from utilities.logger import setup_logger
import re
import base64
import mimetypes

logger = setup_logger(__name__)

def get_system_prompt():
    """Returns the standardized system prompt for all inference engines."""
    return f"""You are a meticulous synthetic data generator that creates high-quality training examples from document sections. Your role is to transform content into question-answer pairs while strictly adhering to these guidelines:

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

def setup_lmdeploy_pipeline(model: str, num_gpus: int = 1, model_format: Optional[str] = None):
    """Initializes and configures the LMDeploy inference pipeline."""
    try:
        nest_asyncio.apply()
        system_prompt = get_system_prompt()
        
        chat_template_config = ChatTemplateConfig('internvl2_5')
        chat_template_config.meta_instruction = system_prompt
        
        backend_config = TurbomindEngineConfig(tp=num_gpus, model_format=model_format)
        
        pipe = pipeline(model, chat_template_config=chat_template_config, backend_config=backend_config)
        
        return pipe
    except ImportError as e:
        logger.error(f"Error importing lmdeploy: {e}")
        logger.error("Please install lmdeploy with: pip install lmdeploy[all]")
        return None
    except Exception as e:
        logger.error(f"Error setting up lmdeploy pipeline: {e}")
        return None

def setup_openrouter_client():
    """Initializes the OpenRouter client using the OpenAI library."""
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        return client
    except Exception as e:
        logger.error(f"Error setting up OpenRouter client: {e}")
        return None

def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image file to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {e}")
        return None

def get_image_mime_type(image_path: str) -> str:
    """Determines the MIME type of an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or 'application/octet-stream'

def call_openrouter_api(
    client: OpenAI,
    prompts: List[Union[str, Dict[str, Any]]],
    model: str,
    max_tokens: int = 4096
) -> List[Any]:
    """Executes batched inference requests using the OpenRouter API with image support."""
    
    class OpenRouterResponse:
        def __init__(self, text):
            self.text = text

    responses = []
    system_prompt = get_system_prompt()

    for i, prompt in enumerate(prompts, 1):
        logger.info(f"Running prompt {i} of {len(prompts)}")
        try:
            content = []
            text_prompt = prompt["text"] if isinstance(prompt, dict) else prompt
            content.append({"type": "text", "text": text_prompt})

            if isinstance(prompt, dict) and "images" in prompt and prompt["images"]:
                for img_path in prompt["images"]:
                    base64_image = encode_image_to_base64(img_path)
                    if base64_image:
                        mime_type = get_image_mime_type(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })

            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=max_tokens,
            )
            
            response_text = completion.choices[0].message.content
            responses.append(OpenRouterResponse(text=response_text))

        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {e}")
            raise
            
    return responses

def call_lmdeploy_api(
    pipe: pipeline,
    prompts: List[Union[str, Dict[str, Any]]],
    max_tokens: int = 4096,
    top_k: int = 40
) -> List[str]:
    """Executes batched inference requests with optional image support.
    
    Args:
        pipe: Initialized LMDeploy pipeline
        prompts: List of prompt strings or dicts with structure:
            {
                "text": "prompt content",
                "images": ["path1.jpg", ...]  # optional
            }
        max_tokens: Maximum number of tokens to generate (default: 4096)
        top_k: Top-k sampling parameter (default: 40)
    
    Returns:
        List of model response strings
    
    Raises:
        ValueError: For malformed prompt structures
        RuntimeError: For API communication failures
    
    Notes:
        - Supports multi-modal prompts (text + images)
        - Images are attached using URL format with dynamic patching
    """

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
    """Extracts the first complete JSON object from a string.
    
    Args:
        string: Input text potentially containing JSON
    
    Returns:
        Extracted JSON string including outer braces
    
    Raises:
        ValueError: If no valid JSON braces are found
    
    Example:
        >>> get_json_str("Prefix {\\"key\\":\\"value\\"} suffix")
        '{\\"key\\":\\"value\\"}'
    """

    first = string.find('{')
    last = string.rfind('}')
    if first == -1 or last == -1 or first > last:
        raise ValueError("Input string does not contain valid JSON object braces.")
    return string[first:last + 1]

def sanitize_json_str(text: str) -> str:
    """Finds all image references in markdown content using {{FIGURE_X}} syntax.
    
    Args:
        section_content: Markdown text to scan
    
    Returns:
        List of found reference names (without braces)
    
    Example:
        >>> extract_image_references("See {{FIGURE_1.2}} and {{TABLE_3}}")
        ['FIGURE_1.2', 'TABLE_3']
    """

    # Remove control characters except \t, \n, \r
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
