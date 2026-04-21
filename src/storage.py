import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def save_to_json(results, filepath="results.json"):
    """Saves the raw structured LLM results to a JSON file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Successfully saved JSON to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")

def save_to_csv(results, filepath="results.csv"):
    """Flattens the structured JSON and saves to a CSV file. Each row is a chunk."""
    rows = []
    
    for source, chunks in results.items():
        for chunk in chunks:
            # Flatten lists to comma-separated strings for CSV brevity
            entities = chunk.get('entities', {})
            people = ", ".join(entities.get('people', []))
            places = ", ".join(entities.get('places', []))
            orgs = ", ".join(entities.get('organizations', []))
            questions = " | ".join(chunk.get('questions', []))
            
            row = {
                "source": source,
                "chunk_index": chunk.get('chunk_index', 0),
                "summary": chunk.get('summary', ""),
                "sentiment": chunk.get('sentiment', ""),
                "confidence_score": chunk.get('confidence_score', 0.0),
                "people_entities": people,
                "place_entities": places,
                "org_entities": orgs,
                "important_questions": questions
            }
            rows.append(row)
            
    try:
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False, encoding='utf-8')
        logger.info(f"Successfully saved CSV to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save CSV to {filepath}: {e}")

def generate_summary_report(results, filepath="summary_report.txt"):
    """Creates a high-level text summary of the run."""
    total_sources = len(results)
    total_chunks = sum(len(chunks) for chunks in results.values())
    
    # Calculate some aggregates
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    avg_confidence_sum = 0
    
    for chunks in results.values():
        for chunk in chunks:
            # Normalize sentiment strings
            sentiment = str(chunk.get('sentiment', 'neutral')).lower()
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            avg_confidence_sum += float(chunk.get('confidence_score', 0.0))
            
    avg_confidence = (avg_confidence_sum / total_chunks) if total_chunks > 0 else 0

    lines = [
        "========================================",
        "      LLM PIPELINE SUMMARY REPORT       ",
        "========================================\n",
        f"Total Inputs Processed: {total_sources}",
        f"Total Chunks Analyzed: {total_chunks}\n",
        "Sentiment Distribution:",
        f"  - Positive: {sentiment_counts['positive']}",
        f"  - Neutral:  {sentiment_counts['neutral']}",
        f"  - Negative: {sentiment_counts['negative']}\n",
        f"Average Confidence Score: {avg_confidence:.2f}\n",
        "Details by Source:"
    ]
    
    for source, chunks in results.items():
        lines.append(f"  - {source}: {len(chunks)} chunks processed successfully.")
        
    lines.append("\n========================================")
    lines.append("End of Report")
    
    report_text = "\n".join(lines)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Successfully generated summary report at {filepath}")
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")
