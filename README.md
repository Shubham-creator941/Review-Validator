# Multi-Agent Review Validation System

An intelligent, parallel multi-agent system built to verify e-commerce product reviews instantly, catching fake reviews and approving genuine ones with high accuracy. 

Built as a technical implementation for Aguken AI.

## Architecture
This system utilizes **LangGraph** to execute three specialized Large Language Model (LLM) agents in parallel, significantly reducing verification time compared to sequential processing. A final Aggregator node compiles the parallel outputs to make a definitive decision.

## Simple Data Flow
```text
Customer posts review
    ↓
┌─────────────────────────────────────────┐
│  Multi-Agent Review Validator (2 sec)   │
├─────────────────────────────────────────┤
│                                         │
│  Reviewer Agent:     ✓ CREDIBLE         │
│  Content Agent:      ✓ AUTHENTIC        │
│  Purchase Agent:     ✓ VERIFIED         │
│                                         │
│  Combined Decision:  ✅ APPROVE (94%)   │
│                                         │
└─────────────────────────────────────────┘
    ↓
Review displayed or Blocked
```

## Tech Stack

• Framework: LangGraph, LangChain
• LLM: Groq (Llama 3.1 8B Instant) for ultra-fast, low-latency reasoning
• Language: Python 3


## Specialized Agents

1. Credibility Agent: Evaluates account age and historical trustworthiness.
2. Content Agent: Analyzes the review text for specific, genuine details versus generic spam patterns.
3. Purchase Agent: Verifies database records to ensure the reviewer actually purchased the product.
4. Master Aggregator: Synthesizes the parallel findings into a final APPROVE or BLOCK verdict with a confidence score.


## Local Setup & Execution
1. Clone this repository.
2. Install the required dependencies:
    pip3 install langgraph langchain langchain-groq python-dotenv
3. Create a .env file in the root directory and add your Groq API Key:
    GROQ_API_KEY=your_api_key_here
3. Run the validation system:
    python3 main.py