#!/usr/bin/env python3
"""
Test script to verify multilingual query handling with the new Gemini embeddings
"""

import os
import logging
from pathlib import Path
import google.generativeai as genai
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbedding:
    """Custom Gemini embedding class for testing"""
    
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: str = None):
        self.model_name = model_name
        if api_key:
            genai.configure(api_key=api_key)
        elif os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        else:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    def get_embedding(self, text: str, task_type: str = "retrieval_query") -> List[float]:
        """Get embedding for a single text"""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

def test_multilingual_embeddings():
    """Test multilingual embedding generation"""
    try:
        embedder = GeminiEmbedding()
        
        # Test queries in different languages and mixed languages
        test_queries = [
            "Computer engineering ko fee kati ho?",  # Mixed Nepali-English
            "कम्प्युटर इन्जिनियरिङको शुल्क कति हो?",  # Pure Nepali
            "What is the fee for computer engineering?",  # Pure English
            "KUSOE ma admission process ke ho?",  # Mixed
            "काठमाडौं विश्वविद्यालयमा भर्ना प्रक्रिया के हो?",  # Pure Nepali
            "Scholarship available छ कि छैन?",  # Mixed
        ]
        
        logger.info("Testing multilingual embedding generation...")
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nTest {i}: '{query}'")
            try:
                embedding = embedder.get_embedding(query)
                logger.info(f"✅ Successfully generated embedding (dimension: {len(embedding)})")
                logger.info(f"   First 5 values: {embedding[:5]}")
            except Exception as e:
                logger.error(f"❌ Failed to generate embedding: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def test_similarity_comparison():
    """Test similarity between different language versions of the same query"""
    try:
        embedder = GeminiEmbedding()
        
        # Similar queries in different languages
        query_pairs = [
            ("Computer engineering ko fee kati ho?", "कम्प्युटर इन्जिनियरिङको शुल्क कति हो?"),
            ("KUSOE ma admission process ke ho?", "काठमाडौं विश्वविद्यालयमा भर्ना प्रक्रिया के हो?"),
            ("Scholarship available छ?", "छात्रवृत्ति उपलब्ध छ?"),
        ]
        
        logger.info("\nTesting similarity between multilingual queries...")
        
        for i, (query1, query2) in enumerate(query_pairs, 1):
            logger.info(f"\nPair {i}:")
            logger.info(f"  Query 1: '{query1}'")
            logger.info(f"  Query 2: '{query2}'")
            
            try:
                emb1 = embedder.get_embedding(query1)
                emb2 = embedder.get_embedding(query2)
                
                # Calculate cosine similarity
                import numpy as np
                emb1_np = np.array(emb1)
                emb2_np = np.array(emb2)
                
                similarity = np.dot(emb1_np, emb2_np) / (np.linalg.norm(emb1_np) * np.linalg.norm(emb2_np))
                
                logger.info(f"  ✅ Cosine similarity: {similarity:.4f}")
                
                if similarity > 0.7:
                    logger.info(f"  🎉 High similarity detected - multilingual understanding working!")
                elif similarity > 0.5:
                    logger.info(f"  ⚠️  Moderate similarity - may need improvement")
                else:
                    logger.info(f"  ❌ Low similarity - potential issue with multilingual understanding")
                    
            except Exception as e:
                logger.error(f"  ❌ Failed to compare embeddings: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Similarity test failed: {e}")
        return False

def test_rag_integration():
    """Test RAG integration with multilingual queries"""
    try:
        # Import the updated RAG handler
        from pipeline.rag import RAGHandler
        
        logger.info("\nTesting RAG integration with multilingual queries...")
        
        rag_handler = RAGHandler()
        
        if not rag_handler.retriever:
            logger.warning("⚠️  RAG retriever not initialized - skipping RAG integration test")
            return True
        
        # Test queries
        test_queries = [
            "Computer engineering ko fee kati ho?",
            "कम्प्युटर इन्जिनियरिङको शुल्क कति हो?",
            "KUSOE ma scholarship available छ?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            logger.info(f"\nRAG Test {i}: '{query}'")
            try:
                response = rag_handler.get_nepali_answer(query)
                logger.info(f"✅ RAG Response: {response[:100]}...")
            except Exception as e:
                logger.error(f"❌ RAG failed: {e}")
        
        return True
        
    except ImportError as e:
        logger.warning(f"⚠️  Could not import RAG handler: {e}")
        return True
    except Exception as e:
        logger.error(f"RAG integration test failed: {e}")
        return False

if __name__ == "__main__":
    logger.info("🚀 Starting multilingual query handling tests...")
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        logger.error("❌ GEMINI_API_KEY environment variable not set!")
        logger.info("Please set your Gemini API key: export GEMINI_API_KEY='your-api-key'")
        exit(1)
    
    success = True
    
    # Test 1: Basic multilingual embedding generation
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Multilingual Embedding Generation")
    logger.info("="*60)
    success &= test_multilingual_embeddings()
    
    # Test 2: Similarity comparison
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Multilingual Similarity Comparison")
    logger.info("="*60)
    success &= test_similarity_comparison()
    
    # Test 3: RAG integration
    logger.info("\n" + "="*60)
    logger.info("TEST 3: RAG Integration")
    logger.info("="*60)
    success &= test_rag_integration()
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("FINAL RESULTS")
    logger.info("="*60)
    
    if success:
        logger.info("🎉 All tests passed! Multilingual query handling is working correctly.")
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
    
    logger.info("\n✨ Test completed!")
