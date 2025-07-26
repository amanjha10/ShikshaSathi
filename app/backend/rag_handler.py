# RAG pipeline with Gemini multilingual embeddings
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

    def get_nepali_answer(self, query: str):
        """
        Retrieves context for the query and returns only the Nepali answer for TTS.
        Use this method to get the final output for TTS, not the raw context.
        """
        context = self.query(query)
        nepali_answer = self.summarize_in_nepali(context, query)
        # Only return the Nepali answer, not the context
        return nepali_answer

    def summarize_in_nepali(self, context: str, query: str):
        """
        Passes the retrieved context and user query to Gemini LLM with a proper RAG prompt for Nepali summarization.
        """
        if not context:
            return "माफ गर्नुहोस्, मसँग यस बारेमा जानकारी छैन। के तपाईं हाम्रो ग्राहक प्रतिनिधिसँग कुरा गर्न चाहनुहुन्छ?"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment")
            return "माफ गर्नुहोस्, थप जानकारी उपलब्ध छैन।"

        from llama_index.llms.gemini import Gemini
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
            "कृपया अंकहरूलाई शब्दमा लेख्नुहोस्, संख्यात्मक अंक प्रयोग नगर्नुहोस्। "
            "यदि सन्दर्भमा प्रश्नको उत्तर छैन भने 'माफ गर्नुहोस्, मसँग यस बारेमा जानकारी छैन। के तपाईं हाम्रो ग्राहक प्रतिनिधिसँग कुरा गर्न चाहनुहुन्छ?' भन्नुहोस्।\n"
            f"\nप्रश्न: {query}\n"
            f"सन्दर्भ:\n{context}"
        )
        try:
            response = llm.complete(prompt)
            return str(response)
        except Exception as e:
            logging.error(f"Error in Gemini LLM summarization: {e}")
            return "माफ गर्नुहोस्, मसँग यस बारेमा जानकारी छैन। के तपाईं हाम्रो ग्राहक प्रतिनिधिसँग कुरा गर्न चाहनुहुन्छ?"

    def _initialize(self):
        try:
            # Use Gemini multilingual embedding model instead of English-only BAAI
            embed_model = GeminiEmbedding(model_name="models/text-embedding-004")
            Settings.embed_model = embed_model
            # Try different possible paths for vector database
            possible_paths = ["../../vector-db", "../vector-db", "vector-db"]
            vector_db_path = None
            for path in possible_paths:
                if Path(path).exists():
                    vector_db_path = Path(path)
                    break

            if not vector_db_path:
                logging.warning(f"Vector database not found in any of {possible_paths}. RAG will be limited.")
                self.retriever = None
                return

            chroma_client = chromadb.PersistentClient(path=str(vector_db_path))
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
            self.retriever = index.as_retriever(similarity_top_k=10)
        except Exception as e:
            logging.error(f"Error initializing RAG system: {e}")
            self.retriever = None

    def query(self, text: str):
        logging.info(f"Backend RAGHandler.query called with text: {text}")
        if not self.retriever:
            logging.error("Backend retriever not initialized. Returning fallback.")
            return None
        try:
            # Expand query with related terms for better retrieval
            expanded_query = self._expand_query(text)
            logging.info(f"Backend expanded query: {expanded_query}")

            logging.info(f"Backend attempting to retrieve nodes for query: {expanded_query}")
            nodes = self.retriever.retrieve(expanded_query)
            logging.info(f"Backend retrieved {len(nodes)} nodes")

            # Filter nodes to find most relevant ones
            relevant_chunks = []
            for i, node in enumerate(nodes):
                if hasattr(node, 'text'):
                    # Check if node is relevant to the query
                    if self._is_relevant_node(node.text, text):
                        relevant_chunks.append(node.text)
                        logging.info(f"Backend Node {i+1} (relevant) text preview: {node.text[:100]}...")
                    else:
                        logging.info(f"Backend Node {i+1} (filtered out) text preview: {node.text[:100]}...")
                else:
                    logging.warning(f"Backend Node {i+1} has no text attribute")

            if relevant_chunks:
                context = '\n---\n'.join(relevant_chunks)
                logging.info(f"Backend RAG retrieved {len(relevant_chunks)} relevant context chunks, total length: {len(context)}")
                return context
            else:
                logging.warning("Backend: No relevant context chunks found")
                return None
        except Exception as e:
            logging.error(f"Backend error in RAG retrieval: {e}")
            import traceback
            logging.error(f"Backend traceback: {traceback.format_exc()}")
            return None

    def _expand_query(self, query: str):
        """Expand query with related terms for better retrieval"""
        query_lower = query.lower()

        # Add related terms based on query content
        expansions = []

        # Computer Engineering related terms
        if any(term in query_lower for term in ['computer', 'कम्प्युटर', 'ce']):
            expansions.extend(['कम्प्युटर इन्जिनियरिङ', 'computer engineering', 'CE'])

        # Fee related terms
        if any(term in query_lower for term in ['fee', 'cost', 'शुल्क', 'फिस', 'पैसा', 'कति']):
            expansions.extend(['शुल्क संरचना', 'fee structure', 'cost', 'फिस'])

        # Mechanical Engineering
        if any(term in query_lower for term in ['mechanical', 'मेकानिकल', 'me']):
            expansions.extend(['मेकानिकल इन्जिनियरिङ', 'mechanical engineering', 'ME'])

        # Civil Engineering
        if any(term in query_lower for term in ['civil', 'सिभिल']):
            expansions.extend(['सिभिल इन्जिनियरिङ', 'civil engineering'])

        # Electrical Engineering
        if any(term in query_lower for term in ['electrical', 'electronics', 'विद्युत', 'इलेक्ट्रोनिक्स', 'ee']):
            expansions.extend(['विद्युत तथा इलेक्ट्रोनिक्स इन्जिनियरिङ', 'electrical electronics engineering', 'EE'])

        # IT
        if any(term in query_lower for term in ['information technology', 'it', 'सूचना प्रविधि']):
            expansions.extend(['सूचना प्रविधि', 'information technology', 'IT'])

        # AI
        if any(term in query_lower for term in ['artificial intelligence', 'ai', 'कृत्रिम बुद्धिमत्ता']):
            expansions.extend(['कृत्रिम बुद्धिमत्ता', 'artificial intelligence', 'AI'])

        if expansions:
            return f"{query} {' '.join(expansions)}"
        return query

    def _is_relevant_node(self, node_text: str, original_query: str):
        """Check if a node is relevant to the original query"""
        query_lower = original_query.lower()
        node_lower = node_text.lower()

        # For fee queries, prioritize nodes with fee information
        if any(term in query_lower for term in ['fee', 'cost', 'शुल्क', 'फिस', 'पैसा', 'कति']):
            if any(term in node_lower for term in ['शुल्क', 'fee', 'cost', 'रुपैयाँ', 'नेपाली रुपैयाँ']):
                # Check if it's the right program
                if any(term in query_lower for term in ['computer', 'कम्प्युटर', 'ce']):
                    return 'कम्प्युटर इन्जिनियरिङ' in node_lower or 'computer engineering' in node_lower
                elif any(term in query_lower for term in ['mechanical', 'मेकानिकल', 'me']):
                    return 'मेकानिकल इन्जिनियरिङ' in node_lower or 'mechanical engineering' in node_lower
                elif any(term in query_lower for term in ['civil', 'सिभिल']):
                    return 'सिभिल इन्जिनियरिङ' in node_lower or 'civil engineering' in node_lower
                elif any(term in query_lower for term in ['electrical', 'electronics', 'विद्युत', 'इलेक्ट्रोनिक्स', 'ee']):
                    return 'विद्युत तथा इलेक्ट्रोनिक्स' in node_lower or 'electrical' in node_lower
                elif any(term in query_lower for term in ['information technology', 'it', 'सूचना प्रविधि']):
                    return 'सूचना प्रविधि' in node_lower or 'information technology' in node_lower
                elif any(term in query_lower for term in ['artificial intelligence', 'ai', 'कृत्रिम बुद्धिमत्ता']):
                    return 'कृत्रिम बुद्धिमत्ता' in node_lower or 'artificial intelligence' in node_lower
                else:
                    return True  # General fee query

        # For general program queries
        if any(term in query_lower for term in ['computer', 'कम्प्युटर', 'ce']):
            return 'कम्प्युटर इन्जिनियरिङ' in node_lower or 'computer engineering' in node_lower

        return True  # Default to including the node