# Customer Support Agent with RAG

This project implements a customer support agent for an e-commerce platform using **Retrieval-Augmented Generation (RAG)**. It leverages a knowledge base stored in text files, vectorized with Chroma, and powered by OpenAI's embeddings and language models to provide accurate, context-aware responses.

## Approach

The agent uses a **RAG** architecture to combine retrieval and generation:
1. **Knowledge Base Loading**: Text files (`products.txt`, `policies.txt`) are loaded from the `knowledge_base` directory, split into chunks by blank lines, and converted into `Document` objects with metadata (category, source).
2. **Vector Store**: Documents are embedded using OpenAI's `text-embedding-ada-002` model and stored in a persistent Chroma vector database (`chroma_db_cs-ai-agent`). This enables efficient similarity-based retrieval.
3. **Retrieval**: The agent's retriever uses similarity search with a threshold (`0.7`) to fetch the top relevant document (`k=1`, where `k` is the number of documents to retrieve) for a user's query.
4. **Generation**: Retrieved context is fed into a `gpt-3.5-turbo` model via a custom `PromptTemplate`, ensuring polite, professional responses. If no relevant context is found, the agent admits it doesn’t know rather than hallucinating.
5. **Chain**: The `RetrievalQA` chain integrates retrieval and generation, returning both the answer and source documents for transparency.

This RAG setup ensures responses are grounded in the knowledge base, minimizing inaccuracies while leveraging the LLM's natural language capabilities.

## Models Used

The RAG implementation relies on the following models from OpenAI:
- **Embedding Model**: `text-embedding-ada-002`  
  - Used to generate vector embeddings for the knowledge base documents, enabling efficient similarity-based retrieval.
- **LLM Model**: `gpt-3.5-turbo`  
  - Powers the generation step, producing polite and professional responses based on retrieved context.

## Setup Instructions
Follow these steps to set up and run the Customer Support Agent:

1. **Prerequisites**:
   - Python 3.8 or higher
   - Required packages: `langchain`, `langchain_openai`, `langchain_chroma`, `chromadb`
   - An OpenAI API key (obtainable from [OpenAI](https://openai.com))

2. **Installation**:
   - Install the required packages with their specific versions as listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt  # Install dependencies

3. **Start the agent**:

    ```bash
    python3 cs-agent.py

### **(Optional) Using a Virtual Environment**

    
    python3 -m venv env          # Create a virutal environment
    source env/bin/activate      # Activate it (use `env\Scripts\activate` on Windows)
    
## Example Conversations
### Example 1
**You:** How much does the UltraPhone X cost?
**Agent:** The UltraPhone X costs $799. It is available in Black, Silver, and Blue colors.

### Example 2
**You:** What are your shipping options?
**Agent:** We offer standard shipping (3-5 business days) for $4.99 and express shipping (1-2 business days) for $12.99. Orders over $50 qualify for free standard shipping.

### Example 3
**You:** Do you have UltraPhone X in stock?
**Agent:** Yes, the UltraPhone X is currently in stock.

---