import numpy as np
import openai
import logging
from typing import List, Dict, Any, Optional
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    A simple vector store implementation using OpenAI embeddings.
    Allows for storing text embeddings and performing similarity searches.
    """
    
    def __init__(self):
        """Initialize an empty vector store."""
        self.embeddings = []
        self.texts = []
        self.metadatas = []
        openai.api_key = settings.OPENAI_API_KEY
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add texts and their metadata to the vector store.
        
        Args:
            texts: List of text strings to embed and store
            metadatas: Optional list of metadata dictionaries corresponding to each text
        """
        if not texts:
            return
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Generate embeddings for all texts
        try:
            embeddings = self._get_embeddings(texts)
            
            # Store texts, embeddings, and metadata
            self.texts.extend(texts)
            self.embeddings.extend(embeddings)
            self.metadatas.extend(metadatas)
            
            logger.info(f"Added {len(texts)} texts to vector store")
        
        except Exception as e:
            logger.error(f"Error adding texts to vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Search for texts similar to the query.
        
        Args:
            query: The query text to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries with keys 'page_content' and 'metadata'
        """
        if not self.embeddings:
            logger.warning("Vector store is empty, returning empty results")
            return []
        
        try:
            # Generate embedding for the query
            query_embedding = self._get_embeddings([query])[0]
            
            # Calculate cosine similarity between query and all stored embeddings
            similarities = self._cosine_similarity(query_embedding, self.embeddings)
            
            # Get indices of top k similar items
            if k > len(similarities):
                k = len(similarities)
            
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                results.append({
                    "page_content": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "similarity": similarities[idx]
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            return []
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for a list of texts using OpenAI API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            # Ensure each text is within token limits (rough approximation)
            processed_texts = [text[:8000] for text in texts]
            
            response = openai.Embedding.create(
                input=processed_texts,
                model="text-embedding-ada-002"  # Use the appropriate embedding model
            )
            
            # Extract embeddings from response
            embeddings = [data["embedding"] for data in response["data"]]
            return embeddings
        
        except Exception as e:
            logger.error(f"Error getting embeddings from OpenAI: {str(e)}")
            # Return zero embeddings as fallback
            dimension = 1536  # OpenAI ada-002 embedding dimension
            return [np.zeros(dimension).tolist() for _ in texts]
    
    def _cosine_similarity(self, query_embedding: List[float], embeddings: List[List[float]]) -> np.ndarray:
        """
        Calculate cosine similarity between a query embedding and a list of embeddings.
        
        Args:
            query_embedding: The query embedding vector
            embeddings: List of embedding vectors to compare against
            
        Returns:
            Array of similarity scores
        """
        query_array = np.array(query_embedding)
        embeddings_array = np.array(embeddings)
        
        # Normalize the vectors
        query_norm = np.linalg.norm(query_array)
        embeddings_norm = np.linalg.norm(embeddings_array, axis=1)
        
        # Avoid division by zero
        if query_norm == 0:
            query_norm = 1e-10
        embeddings_norm = np.where(embeddings_norm == 0, 1e-10, embeddings_norm)
        
        # Calculate dot product
        dot_product = np.dot(embeddings_array, query_array)
        
        # Calculate cosine similarity
        similarities = dot_product / (embeddings_norm * query_norm)
        
        return similarities 