import os
import argparse
import logging
from dotenv import load_dotenv

from src.ingestion import ingest_all
from src.preprocessing import preprocess_documents
from src.llm_client import process_documents_with_llm
from src.storage import save_to_json, save_to_csv, generate_summary_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Entity Extraction and Analysis Pipeline")
    parser.add_argument(
        '--inputs',
        nargs='+',
        help="List of input file paths (.txt, .pdf) and/or URLs.",
        required=True
    )
    return parser.parse_args()

def main():
    load_dotenv()
    
   
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_gemini_api_key_here":
        logger.error("Please set GEMINI_API_KEY in the .env file.")
        return

    args = parse_args()
    inputs = args.inputs
    
    logger.info(f"Starting pipeline with {len(inputs)} inputs...")
    
    # 1. Ingestion
    logger.info("--- Step 1: Ingestion ---")
    raw_data = ingest_all(inputs)
    if not raw_data:
        logger.error("No data could be ingested from the provided inputs. Exiting.")
        return
        
   
    logger.info("--- Step 2: Preprocessing ---")
    processed_data = preprocess_documents(raw_data)
    
   
    logger.info("--- Step 3: LLM Analysis ---")
    results = process_documents_with_llm(processed_data)
    
    
    logger.info("--- Step 4: Storing Results ---")
    save_to_json(results, os.path.join(os.getcwd(), 'results.json'))
    save_to_csv(results, os.path.join(os.getcwd(), 'results.csv'))
    generate_summary_report(results, os.path.join(os.getcwd(), 'summary_report.txt'))
    
    logger.info("Pipeline execution complete! Check results.json, results.csv, and summary_report.txt.")

if __name__ == "__main__":
    main()
