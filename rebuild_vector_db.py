#!/usr/bin/env python3
"""
Script to rebuild the vector database with Nepali content using Gemini embeddings
"""

import os
import logging
import chromadb
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.embeddings import BaseEmbedding
import google.generativeai as genai
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbedding(BaseEmbedding):
    """Custom Gemini embedding class for LlamaIndex"""
    
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str = None):
        super().__init__()
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for a single query"""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting text embedding: {e}")
            raise
    
    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embeddings.append(self._get_text_embedding(text))
        return embeddings

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """Async version of _get_query_embedding"""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Async version of _get_text_embedding"""
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Async version of _get_text_embeddings"""
        return self._get_text_embeddings(texts)

def load_documents_from_kusoe_database():
    """Load all documents from KUSOE_database directory"""
    documents = []
    # Try different possible paths for KUSOE_database
    possible_paths = ["KUSOE_database", "../KUSOE_database", "../../KUSOE_database"]
    kusoe_path = None
    for path in possible_paths:
        if Path(path).exists():
            kusoe_path = Path(path)
            break

    if not kusoe_path:
        print(f"KUSOE_database not found in any of {possible_paths}")
        return documents
    
    # Load overview file
    overview_file = kusoe_path / "overview.txt"
    if overview_file.exists():
        with open(overview_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by chunk markers
            chunks = content.split('-c-h-u-n-k-h-e-r-e-')
            for i, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if chunk:
                    doc = Document(
                        text=chunk,
                        metadata={
                            "source": "overview.txt",
                            "chunk_id": i,
                            "type": "overview"
                        }
                    )
                    documents.append(doc)
    
    # Load program files
    programs_path = kusoe_path / "programs"
    if programs_path.exists():
        for program_file in programs_path.glob("*.txt"):
            with open(program_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Split by chunk markers
                chunks = content.split('-c-h-u-n-k-h-e-r-e-')
                for i, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    if chunk:
                        doc = Document(
                            text=chunk,
                            metadata={
                                "source": program_file.name,
                                "chunk_id": i,
                                "type": "program",
                                "program": program_file.stem
                            }
                        )
                        documents.append(doc)
    
    logger.info(f"Loaded {len(documents)} documents from KUSOE database")
    return documents

def rebuild_vector_database():
    """Rebuild the vector database with Nepali content and Gemini embeddings"""
    try:
        # Initialize Gemini embedding model
        embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
        Settings.embed_model = embed_model
        
        # Load documents
        documents = load_documents_from_kusoe_database()
        
        if not documents:
            logger.error("No documents found to index")
            return False
        
        # Remove existing vector database
        vector_db_path = Path("vector-db")
        if vector_db_path.exists():
            import shutil
            shutil.rmtree(vector_db_path)
            logger.info("Removed existing vector database")
        
        # Create new ChromaDB client
        chroma_client = chromadb.PersistentClient(path="vector-db")
        
        # Create new collection
        collection_name = "kusoe_rag_llm"
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass  # Collection might not exist
        
        chroma_collection = chroma_client.create_collection(collection_name)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index from documents
        logger.info("Creating vector index with Gemini embeddings...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        logger.info(f"Successfully rebuilt vector database with {len(documents)} documents")
        logger.info(f"Collection '{collection_name}' created with {chroma_collection.count()} embeddings")
        
        return True
        
    except Exception as e:
        logger.error(f"Error rebuilding vector database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting vector database rebuild...")
    success = rebuild_vector_database()
    if success:
        logger.info("Vector database rebuild completed successfully!")
    else:
        logger.error("Vector database rebuild failed!")
