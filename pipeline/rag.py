"""
RAGHandler: Handles Retrieval-Augmented Generation operations
"""
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.core.embeddings import BaseEmbedding
import chromadb
import os
import logging
from pathlib import Path
import google.generativeai as genai
from typing import List

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
            logging.error(f"Error getting query embedding: {e}")
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
            logging.error(f"Error getting text embedding: {e}")
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

class RAGHandler:
    def __init__(self):
        self.retriever = None
        self._initialize()

    def _initialize(self):
        try:
            # Use Gemini multilingual embedding model instead of English-only BAAI
            embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
            Settings.embed_model = embed_model
            vector_db_path = Path("vector-db/")
            if not vector_db_path.exists():
                logging.warning(f"Vector database not found at {vector_db_path}. RAG will be limited.")
                self.retriever = None
                return
            chroma_client = chromadb.PersistentClient(path="vector-db/")
            collections = chroma_client.list_collections()
            if not collections:
                logging.warning("No collections found in ChromaDB. RAG will be limited.")
                self.retriever = None
                return
            # Get the collection directly
            try:
                chroma_collection = chroma_client.get_collection('kusoe_rag_llm')
            except Exception as e:
                logging.warning(f"'kusoe_rag_llm' collection not found in ChromaDB: {e}. RAG will be limited.")
                self.retriever = None
                return
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            self.retriever = index.as_retriever(similarity_top_k=3)
        except Exception as e:
            logging.error(f"Error initializing RAG system: {e}")
            self.retriever = None

    def query(self, text: str):
        logging.info(f"RAGHandler.query called with text: {text}")
        if not self.retriever:
            logging.error("Retriever not initialized. Returning fallback.")
            return None
        try:
            logging.info(f"Attempting to retrieve nodes for query: {text}")
            nodes = self.retriever.retrieve(text)
            logging.info(f"Retrieved {len(nodes)} nodes")
            context_chunks = []
            for i, node in enumerate(nodes):
                if hasattr(node, 'text'):
                    context_chunks.append(node.text)
                    logging.info(f"Node {i+1} text preview: {node.text[:100]}...")
                else:
                    logging.warning(f"Node {i+1} has no text attribute")

            if context_chunks:
                context = '\n---\n'.join(context_chunks)
                logging.info(f"RAG retrieved {len(context_chunks)} context chunks, total length: {len(context)}")
                return context
            else:
                logging.warning("No context chunks found")
                return None
        except Exception as e:
            logging.error(f"Error in RAG retrieval: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_nepali_answer(self, query: str):
        """
        Retrieves context for the query and returns only the Nepali answer for TTS.
        Use this method to get the final output for TTS, not the raw context.
        """
        logging.info(f"get_nepali_answer called with query: {query}")
        context = self.query(query)
        logging.info(f"Retrieved context: {context[:200] if context else 'None'}...")
        nepali_answer = self.summarize_in_nepali(context, query)
        logging.info(f"Generated Nepali answer: {nepali_answer}")
        # Only return the Nepali answer, not the context
        return nepali_answer

    def summarize_in_nepali(self, context: str, query: str):
        """
        Passes the retrieved context and user query to Gemini LLM with a proper RAG prompt for Nepali summarization.
        """
        if not context:
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment")
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"

        llm = Gemini(
            model_name="models/gemini-2.5-flash",
            temperature=0.1,
            max_output_tokens=256,
            top_p=0.8,
            top_k=20
        )
        prompt = (
            "तपाईंलाई तलको सन्दर्भ र प्रयोगकर्ताको प्रश्न दिइएको छ। "
            "सन्दर्भको आधारमा छोटो र संक्षिप्त रूपमा नेपालीमा उत्तर दिनुहोस्। "
            "कृपया अंकहरूलाई शब्दमा लेख्नुहोस्, संख्यात्मक अंक प्रयोग नगर्नुहोस्।\n"
            f"\nप्रश्न: {query}\n"
            f"सन्दर्भ:\n{context}"
        )
        try:
            response = llm.complete(prompt)
            return str(response)
        except Exception as e:
            logging.error(f"Error in Gemini LLM summarization: {e}")
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"
