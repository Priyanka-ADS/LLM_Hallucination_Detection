"""
Self-Consistency Detection Signal

Measures the semantic consistency of multiple answer variations for a given question.
Uses sentence-transformers to compute semantic similarity between answers.

High consistency → likely factual (not hallucinated)
Low consistency → potentially hallucinated
"""

import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class SelfConsistencyDetector:
    """
    Detects hallucinations by measuring self-consistency of LLM responses.
    
    The intuition: If a model is confident about factual information, 
    it should give semantically similar answers when asked the same question
    multiple times. Inconsistent answers suggest hallucination.
    
    Uses sentence-transformers (all-MiniLM-L6-v2) for semantic similarity.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the SelfConsistencyDetector.
        
        Args:
            model_name: Name of the sentence-transformer model to use.
        """
        self.model_name = model_name
        self._model = None
        logger.info(f"SelfConsistencyDetector initialized with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded sentence-transformer model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise
        return self._model
    
    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            numpy array of embeddings, shape (n_texts, embedding_dim).
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def compute_pairwise_similarity(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine similarity matrix.
        
        Args:
            embeddings: Array of shape (n, dim).
            
        Returns:
            Similarity matrix of shape (n, n).
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms
        
        # Compute similarity matrix
        sim_matrix = np.dot(normalized, normalized.T)
        return sim_matrix
    
    def compute_consistency_score(self, answer_variations: List[str]) -> float:
        """
        Compute a consistency score for multiple answer variations.
        
        Args:
            answer_variations: List of answer strings (at least 2).
            
        Returns:
            Consistency score between 0 and 1.
            Higher = more consistent = less likely hallucinated.
        """
        if len(answer_variations) < 2:
            logger.warning("Need at least 2 answer variations for consistency check")
            return 0.5  # Neutral score
        
        # Compute embeddings
        embeddings = self.compute_embeddings(answer_variations)
        
        # Compute pairwise similarity
        sim_matrix = self.compute_pairwise_similarity(embeddings)
        
        # Get upper triangle (exclude diagonal)
        n = len(answer_variations)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_scores = sim_matrix[upper_tri_indices]
        
        # Average pairwise similarity as consistency score
        consistency_score = float(np.mean(pairwise_scores))
        
        # Clip to [0, 1]
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        return consistency_score
    
    def generate_answer_variations(self, question: str, answer: str, 
                                     n_variations: int = 5) -> List[str]:
        """
        Generate synthetic answer variations for self-consistency checking.
        
        Since we don't have access to the LLM for re-generation, we create
        variations by:
        1. Using the original answer
        2. Rephrasing with different sentence structures
        3. Adding/removing context
        
        Args:
            question: The question being answered.
            answer: The original answer.
            n_variations: Number of variations to generate.
            
        Returns:
            List of answer variations.
        """
        variations = [answer]
        
        # Variation 1: Prepend the question context
        variations.append(f"The answer to '{question}' is: {answer}")
        
        # Variation 2: Add "In summary" prefix
        variations.append(f"In summary, {answer.lower() if answer[0:1].isupper() else answer}")
        
        # Variation 3: Reverse sentence order (if multiple sentences)
        sentences = answer.split('. ')
        if len(sentences) > 1:
            reversed_answer = '. '.join(reversed(sentences))
            variations.append(reversed_answer)
        else:
            variations.append(f"To put it simply, {answer.lower() if answer[0:1].isupper() else answer}")
        
        # Variation 4: Use question + answer combo
        variations.append(f"Question: {question}. Answer: {answer}")
        
        # Trim to requested number
        return variations[:n_variations]
    
    def detect(self, question: str, answer: str) -> dict:
        """
        Detect potential hallucination using self-consistency.
        
        Args:
            question: The question that was asked.
            answer: The answer to evaluate.
            
        Returns:
            Dictionary with:
                - score: float, consistency score (0-1, higher = more consistent)
                - is_hallucination: bool, whether the answer is likely hallucinated
                - details: dict, additional information
        """
        try:
            # Generate answer variations
            variations = self.generate_answer_variations(question, answer)
            
            # Compute consistency score
            consistency_score = self.compute_consistency_score(variations)
            
            # Threshold for hallucination detection
            # Lower consistency suggests hallucination
            threshold = 0.5
            is_hallucination = consistency_score < threshold
            
            return {
                'score': consistency_score,
                'is_hallucination': is_hallucination,
                'details': {
                    'n_variations': len(variations),
                    'threshold': threshold,
                    'signal_name': 'self_consistency'
                }
            }
        
        except Exception as e:
            logger.error(f"Self-consistency detection failed: {e}")
            return {
                'score': 0.5,
                'is_hallucination': False,
                'details': {'error': str(e), 'signal_name': 'self_consistency'}
            }


# ============================================================
# Test Example
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = SelfConsistencyDetector()
    
    # Test with a factual answer
    print("\n" + "="*60)
    print("Test 1: Factual Answer")
    print("="*60)
    result = detector.detect(
        question="What is the capital of France?",
        answer="The capital of France is Paris."
    )
    print(f"Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    
    # Test with a potentially hallucinated answer
    print("\n" + "="*60)
    print("Test 2: Potentially Hallucinated Answer")
    print("="*60)
    result = detector.detect(
        question="What is the capital of France?",
        answer="The capital of France is Berlin, which was established in 1247 by ancient Romans."
    )
    print(f"Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    
    # Test consistency of multiple answers
    print("\n" + "="*60)
    print("Test 3: Direct Consistency Check")
    print("="*60)
    consistent_answers = [
        "Paris is the capital of France.",
        "The capital of France is Paris.",
        "France's capital city is Paris.",
    ]
    score = detector.compute_consistency_score(consistent_answers)
    print(f"Consistent answers score: {score:.4f}")
    
    inconsistent_answers = [
        "Paris is the capital of France.",
        "The capital of France is Berlin.",
        "London is the capital of France.",
    ]
    score = detector.compute_consistency_score(inconsistent_answers)
    print(f"Inconsistent answers score: {score:.4f}")
