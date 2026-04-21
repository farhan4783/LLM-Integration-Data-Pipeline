import re
import tiktoken
import logging

logger = logging.getLogger(__name__)

def clean_text(text):
    """
    Remove excessive whitespace and normalize the text.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple blank lines with a single blank line
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def chunk_text(text, max_tokens=2000, overlap_tokens=200):
    """
    Splits text into chunks of roughly max_tokens using tiktoken.
    This ensures we comfortably fit within the LLM's context limit.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
    except Exception as e:
        logger.warning(f"Tiktoken encoding failed: {e}. Falling back to character-based chunking.")
        return fallback_chunk_text(text)

    chunks = []
    start = 0
    total_tokens = len(tokens)
    
    while start < total_tokens:
        end = min(start + max_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        # Advance by max_tokens minus overlap, but ensure we always move forward
        step = max_tokens - overlap_tokens
        if step <= 0:
            step = max_tokens  # safety guard
        start += step

    return chunks

def fallback_chunk_text(text, max_chars=8000, overlap_chars=800):
    """
    A crude fallback that splits by character length, roughly approximating token limits.
    """
    chunks = []
    start = 0
    total_chars = len(text)
    
    while start < total_chars:
        end = min(start + max_chars, total_chars)
        # Try to find a nice breaking point (e.g. paragraph or sentence end)
        if end < total_chars:
            break_idx = text.rfind('\n', start, end)
            if break_idx == -1 or break_idx <= start:
                break_idx = text.rfind('. ', start, end)
            
            if break_idx != -1 and break_idx > start + (max_chars // 2):
                end = break_idx + 1
                
        chunk = text[start:end]
        chunks.append(chunk)
        
        step = (end - start) - overlap_chars
        if step <= 0:
            step = end - start
        start += step
        
    return chunks

def preprocess_documents(raw_data):
    """
    Cleans and chunks documents in the raw_data dictionary.
    Returns a dictionary of source -> list of text chunks.
    """
    processed_data = {}
    for source, text in raw_data.items():
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned)
        processed_data[source] = chunks
        logger.info(f"Processed {source} into {len(chunks)} chunks.")
    return processed_data
