import os
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_text_file(filepath):
    """Read a raw text file handling potential encoding issues."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decode failed for {filepath}. Trying latin-1.")
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read text file {filepath}: {e}")
            return ""
    except Exception as e:
        logger.error(f"Failed to read text file {filepath}: {e}")
        return ""

def read_pdf_file(filepath):
    """Read text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to read PDF file {filepath}: {e}")
        return ""

def fetch_url(url, timeout=10):
    """Fetch text from a URL and remove boilerplate."""
    try:
        # Use a generic User-Agent to avoid simple blocks
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            script.decompose()
        
        # Get text and clean it up slightly
        text = soup.get_text(separator=' ')
        return text
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for URL {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Failed to parse URL {url}: {e}")
        return ""

def ingest_all(inputs):
    """
    Given a list of inputs (file paths or URLs), extract text from each.
    Returns a dictionary mapping input source -> raw text.
    """
    raw_data = {}
    for item in inputs:
        source = item.strip()
        if source.startswith('http://') or source.startswith('https://'):
            logger.info(f"Ingesting URL: {source}")
            text = fetch_url(source)
        elif os.path.isfile(source):
            _, ext = os.path.splitext(source)
            if ext.lower() == '.pdf':
                logger.info(f"Ingesting PDF: {source}")
                text = read_pdf_file(source)
            elif ext.lower() == '.txt':
                logger.info(f"Ingesting TXT: {source}")
                text = read_text_file(source)
            else:
                logger.warning(f"Unsupported file extension for {source}")
                continue
        else:
            logger.warning(f"Input not recognized or file not found: {source}")
            continue
            
        if text.strip():
            raw_data[source] = text
        else:
            logger.warning(f"No text extracted from {source}")

    return raw_data
