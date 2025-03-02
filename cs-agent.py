import os

from langchain.docstore.document import Document
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from typing import List, Optional
from langchain.prompts import PromptTemplate


# Configuration constants (could be moved to env vars or config file)
BASE_DIR = Path("knowledge_base")
PERSIST_DIRECTORY = "chroma_db_cs-ai-agent"
EMBEDDING_MODEL = "text-embedding-ada-002"
LLM_MODEL = "gpt-3.5-turbo"
SIMILARITY_THRESHOLD = 0.7
TOP_K = 1


def load_knowledge_base(file_path: Path, category: str) -> List[Document]:
    """Load text file, split by blank lines, and convert to Document objects."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            items = file.read().strip().split("\n\n")
            return [Document(page_content=item.strip(), metadata={"category": category, "source": str(file_path)}) 
                   for item in items if item.strip()]
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return []

def setup_vector_store(documents: List[Document], persist_directory: str, 
                      embeddings: OpenAIEmbeddings) -> Optional[Chroma]:
    """Set up or load Chroma vector store."""
    try:
        if not documents:
            print("No documents provided for vector store")
            return None
            
        if os.path.exists(persist_directory):
            print("Loading existing Chroma DB from disk...")
            return Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        
        print("Creating new Chroma DB...")
        return Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist_directory)
    except Exception as e:
        print(f"Failed to setup vector store: {str(e)}")
        return None

def setup_chain(vector_store: Chroma, query: str, llm: ChatOpenAI, 
                threshold: float = SIMILARITY_THRESHOLD) -> Optional[RetrievalQA]:
    """Set up the chain with similarity search."""
    try:
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={ "k": TOP_K, "score_threshold": threshold}
        )
        
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a polite and professional customer support agent. Use the provided context to answer the question. "
                "If the context doesn't provide enough information, just say that you don't know, don't try to make up an answer. "
                "**Context:**\n{context}\n\n"
                "**Question:**\n{question}\n\n"
                "**Answer:**"
            )           
        )               
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
    except Exception as e:
        print(f"Error setting up chain: {str(e)}")
        return None

def run_agent() -> None:
    """Main function to run the e-commerce customer support agent."""
    # Verify API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set")
        return

    # Initialize components
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)
        llm = ChatOpenAI(temperature=0, model_name=LLM_MODEL, openai_api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize embeddings or LLM: {str(e)}")
        return

    # Load knowledge base
    documents = []
    for file_name, category in [("products.txt", "product"), ("policies.txt", "policies")]:
        docs = load_knowledge_base(BASE_DIR / file_name, category)
        documents.extend(docs)
    
    if not documents:
        print("No documents loaded from knowledge base")
        return

    vector_store = setup_vector_store(documents, PERSIST_DIRECTORY, embeddings)
    if not vector_store:
        return

    print("Welcome to Customer Support Agent! How may I assist you today? (Type 'exit' to quit)")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() == "exit":
                print("Agent: Thank you for using our Customer Support. Have a great day!")
                break
                
            if not query:
                print("Agent: Hello! How can I assist you today?")
                continue
                
            chain = setup_chain(vector_store, query, llm)
            if not chain:
                print("Agent: I’m sorry, I couldn’t find the information you’re looking for. ")
                continue
                
            result = chain.invoke({"query": query})
            response = result['result']
            
            print(f"Agent: {response}")
            
        except KeyboardInterrupt:
            print("\nAgent: Thank you for using our Customer Support. Goodbye!")
            break
        except Exception as e:
            # logger.error(f"Error processing query: {str(e)}")
            print("Agent: I apologize, an error occurred while processing your request. Please try again.")

if __name__ == "__main__":
    run_agent()