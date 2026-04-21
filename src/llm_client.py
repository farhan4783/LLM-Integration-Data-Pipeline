import os
import json
import logging
import google.generativeai as genai
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

logger = logging.getLogger(__name__)

# System instructions to enforce JSON output structure
SYSTEM_PROMPT = """
You are an expert entity extraction and text analysis AI. 
You must analyze the provided text and output ONLY a valid JSON object with the following exact structure, no markdown parsing or backticks around it:

{
  "summary": "A 2 to 3 sentence summary of the text.",
  "entities": {
    "people": ["List", "of", "people"],
    "places": ["List", "of", "places"],
    "organizations": ["List", "of", "organizations"]
  },
  "sentiment": "positive", 
  "confidence_score": 0.95,
  "questions": [
    "Important question 1?",
    "Important question 2?",
    "Important question 3?"
  ]
}

Note: sentiment MUST be "positive", "neutral", or "negative". confidence_score MUST be a float between 0.0 and 1.0. If you cannot find entities for a category, use an empty array.
"""

def setup_llm():
    """Initializes the Gemini client if API key is present."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.error("GEMINI_API_KEY environment variable not set.")
        return False
    
    genai.configure(api_key=api_key)
    return True

# Ensure we retry on common API issues
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def _call_gemini_api(model, text):
    response = model.generate_content(
        text,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
        )
    )
    return response.text

def process_chunk_with_llm(text_chunk):
    """
    Submits a chunk of text to Gemini, robustly requesting JSON,
    and handling potential API errors and malformed JSON.
    """
    if not setup_llm():
        return None
        
    # We use gemini-flash-latest as it is fast, cost-effective and perfectly capable of entity extraction.
    model = genai.GenerativeModel(
        model_name='gemini-flash-latest',
        system_instruction=SYSTEM_PROMPT
    )
    
    try:
        raw_output = _call_gemini_api(model, text_chunk)
    except Exception as e:
        logger.error(f"LLM API call failed after retries: {e}")
        return None

    # Parse and validate the JSON
    parsed_json = _parse_json_robustly(raw_output)
    if not parsed_json:
        logger.warning("Failed to parse LLM output as JSON.")
        return None
        
    return parsed_json

def _parse_json_robustly(text_output):
    """
    Attempts to parse JSON from the string, handling potential
    markdown code blocks that LLMs sometimes insert despite instructions.
    """
    # Simply strip markdown boundaries if present
    text = text_output.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
        
    if text.endswith("```"):
        text = text[:-3]
        
    text = text.strip()
    
    try:
        data = json.loads(text)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        logger.debug(f"Raw output attempted to parse: {text_output}")
        # Try to save the situation by finding { and }
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                return json.loads(text[start_idx:end_idx+1])
        except Exception:
            return None
    return None

def process_documents_with_llm(processed_data):
    """
    Iterates through all sources and chunks, calling the LLM and collecting results.
    Return structure: {source: [chunk_result_1, chunk_result_2, ...]}
    """
    results = {}
    
    for source, chunks in processed_data.items():
        logger.info(f"LLM processing for {source} ({len(chunks)} chunks)")
        source_results = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Processing chunk {i+1}/{len(chunks)}...")
            result = process_chunk_with_llm(chunk)
            if result:
                # Keep track of which chunk generated these results
                result['chunk_index'] = i 
                source_results.append(result)
            else:
                logger.warning(f"  Chunk {i+1} failed to yield valid results. Skipping.")
                
        results[source] = source_results
        
    return results
