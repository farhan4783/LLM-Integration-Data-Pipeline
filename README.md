# LLM Entity Extraction and Analysis Pipeline

This repository contains a Python application that ingests unstructured text data from multiple sources (PDFs, TXT files, and URLs), preprocesses it, chunks the text, and extracts structured insights using the Gemini LLM API. The pipeline is robust to failures and built strictly without orchestration frameworks like LangChain.

## Features
- **Multi-source Ingestion**: Parses `.txt`, `.pdf` and fetches body content from website URLs while stripping out boilerplate.
- **Smart Preprocessing**: Uses local token counting via `tiktoken` to dynamically chunk text into digestible segments that fit well within context windows.
- **Robust LLM Interfacing**: Interacts directly with the Gemini API (`google-generativeai`), heavily enforcing a precise JSON schema response. Retries gracefully upon timeouts and API limits using `tenacity`.
- **Structured Storage**: Drops all chunks into `results.json`, flattens data into `results.csv`, and aggregates metrics into a high-level `summary_report.txt`. 
- **Graceful Failures**: A single faulty URL or unparseable chunk will not crash the entire pipeline; errors are logged, and execution continues.

## Project Structure
```text
llm_pipeline/
│
├── main.py
├── requirements.txt
├── .env                  
├── README.md
│
└── src/
    ├── ingestion.py      (Handles PDF, TXT, and Web scraping)
    ├── preprocessing.py  (Cleans and chunks text)
    ├── llm_client.py     (Handles API calls, retries, and strict JSON parsing)
    └── storage.py        (Saves JSON, CSV, and Summary)
```

## Setup Instructions

**1. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Configure your Environment Variables**
Configure your Gemini API key. Ensure you have an `.env` file in the root directory:
```text
GEMINI_API_KEY=your_actual_api_key_here
```

## Execution

Run `main.py` passing the target inputs via the `--inputs` flag.

```bash
python main.py --inputs "sample.txt" "https://en.wikipedia.org/wiki/Large_language_model" "https://invalidurl.this.will.fail.com"
```

The pipeline will log its progress. Upon completion, the following files are produced in the current directory:
- `results.json`: Full nested schema output from the LLM.
- `results.csv`: Flattened tabular format suited for Excel.
- `summary_report.txt`: Aggregate metrics (sentiment distributions, confidence).

## Design Decisions and Known Limitations
- **Chunking with Tiktoken**: I utilized OpenAI's `tiktoken` locally purely as an effective tokenizer approximation so we don't have to guess chunk boundaries. Character-based chunking is provided as a fallback.
- **Prompt vs Structural JSON Enforcements**: I enforced the JSON output within the System Prompt rather than the newer native features to maintain broader compatibility and strictly handle malformed markdown boundary injections manually (`_parse_json_robustly`). 
- **No Async (Yet)**: The system fetches inputs and parses chunks synchronously relative to each source, ensuring clean traceability in logs. To drastically increase speed as inputs soar, moving to `asyncio` & `httpx` in a future version is recommended.
- **Error Types Supported**: Network timeouts (on scrape or LLM call), JSON decode errors, Token limit triggers, URL resolution failures.

## Demonstration (Video)

<video src="Recording.mp4" controls width="100%" style="max-width: 800px;"></video>




