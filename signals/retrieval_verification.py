"""
Retrieval Verification Signal

Verifies LLM answers against a knowledge corpus using:
1. BM25 for document retrieval
2. NLI (Natural Language Inference) for entailment checking

High entailment score → answer is supported by evidence → less likely hallucinated
Low entailment score → answer contradicts or is not supported → possibly hallucinated
"""

import os
import pickle
import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class RetrievalVerifier:
    """
    Verifies answers against a knowledge corpus using BM25 retrieval 
    and NLI-based entailment checking.
    
    Pipeline:
    1. Given a question, retrieve top-K relevant documents using BM25
    2. Check if the answer is entailed by the retrieved evidence
    3. Return an entailment score (0-1)
    """
    
    def __init__(self, corpus_dir: str = 'data/corpus', 
                 nli_model: str = 'facebook/bart-large-mnli'):
        """
        Initialize the RetrievalVerifier.
        
        Args:
            corpus_dir: Directory containing the corpus files.
            nli_model: Name of the NLI model for entailment checking.
        """
        self.corpus_dir = corpus_dir
        self.nli_model_name = nli_model
        self._nli_pipeline = None
        self._documents = None
        self._bm25 = None
        self._tokenized_docs = None
        
        # Load corpus if available
        self._load_corpus()
        
        logger.info(f"RetrievalVerifier initialized")
    
    def _load_corpus(self):
        """Load the pre-built corpus and BM25 index."""
        try:
            docs_path = os.path.join(self.corpus_dir, 'documents.pkl')
            bm25_path = os.path.join(self.corpus_dir, 'bm25_index.pkl')
            tokens_path = os.path.join(self.corpus_dir, 'tokenized_docs.pkl')
            
            if os.path.exists(docs_path) and os.path.exists(bm25_path):
                with open(docs_path, 'rb') as f:
                    self._documents = pickle.load(f)
                with open(bm25_path, 'rb') as f:
                    self._bm25 = pickle.load(f)
                if os.path.exists(tokens_path):
                    with open(tokens_path, 'rb') as f:
                        self._tokenized_docs = pickle.load(f)
                logger.info(f"Loaded corpus with {len(self._documents)} documents")
            else:
                logger.warning(f"Corpus files not found in {self.corpus_dir}. "
                             "Run prepare_knowledge_corpus.py first.")
        except Exception as e:
            logger.error(f"Failed to load corpus: {e}")
    
    @property
    def nli_pipeline(self):
        """Lazy load the NLI pipeline."""
        if self._nli_pipeline is None:
            try:
                from transformers import pipeline
                self._nli_pipeline = pipeline(
                    'zero-shot-classification',
                    model=self.nli_model_name,
                    device=-1  # CPU
                )
                logger.info(f"Loaded NLI model: {self.nli_model_name}")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                raise
        return self._nli_pipeline
    
    def _tokenize_query(self, query: str) -> List[str]:
        """Tokenize a query string."""
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            return word_tokenize(query.lower())
        except Exception:
            return query.lower().split()
    
    def retrieve_evidence(self, question: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-K relevant documents for a given question.
        
        Args:
            question: The question to search for.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of dictionaries with document info and relevance scores.
        """
        if self._bm25 is None or self._documents is None:
            logger.warning("No corpus loaded. Returning empty evidence.")
            return []
        
        # Tokenize query
        query_tokens = self._tokenize_query(question)
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-K indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if idx < len(self._documents):
                doc = self._documents[idx]
                results.append({
                    'text': doc['text'],
                    'title': doc.get('title', ''),
                    'score': float(scores[idx]),
                    'doc_id': doc.get('doc_id', str(idx))
                })
        
        return results
    
    def verify_answer(self, answer: str, evidence: List[Dict]) -> Dict:
        """
        Verify if an answer is supported by retrieved evidence using NLI.
        
        Args:
            answer: The answer to verify.
            evidence: List of evidence documents.
            
        Returns:
            Dictionary with entailment scores and details.
        """
        if not evidence:
            return {
                'entailment_score': 0.5,
                'contradiction_score': 0.0,
                'neutral_score': 0.5,
                'best_evidence': None,
                'details': 'No evidence available'
            }
        
        entailment_scores = []
        contradiction_scores = []
        neutral_scores = []
        evidence_results = []
        
        for doc in evidence:
            premise = doc['text']
            
            # Use NLI to check entailment
            try:
                result = self.nli_pipeline(
                    premise,
                    candidate_labels=['entailment', 'contradiction', 'neutral'],
                    hypothesis=answer,
                    multi_label=False
                )
                
                # Extract scores
                label_scores = dict(zip(result['labels'], result['scores']))
                ent_score = label_scores.get('entailment', 0.0)
                con_score = label_scores.get('contradiction', 0.0)
                neu_score = label_scores.get('neutral', 0.0)
                
                entailment_scores.append(ent_score)
                contradiction_scores.append(con_score)
                neutral_scores.append(neu_score)
                
                evidence_results.append({
                    'text': premise[:200],
                    'entailment': ent_score,
                    'contradiction': con_score,
                    'neutral': neu_score
                })
                
            except Exception as e:
                logger.warning(f"NLI check failed for evidence: {e}")
                continue
        
        if not entailment_scores:
            return {
                'entailment_score': 0.5,
                'contradiction_score': 0.0,
                'neutral_score': 0.5,
                'best_evidence': None,
                'details': 'NLI processing failed'
            }
        
        # Use max entailment score (best supporting evidence)
        max_ent_idx = np.argmax(entailment_scores)
        
        return {
            'entailment_score': float(np.max(entailment_scores)),
            'avg_entailment_score': float(np.mean(entailment_scores)),
            'contradiction_score': float(np.mean(contradiction_scores)),
            'neutral_score': float(np.mean(neutral_scores)),
            'best_evidence': evidence_results[max_ent_idx] if evidence_results else None,
            'n_evidence_checked': len(entailment_scores)
        }
    
    def detect(self, question: str, answer: str, top_k: int = 5) -> dict:
        """
        Detect potential hallucination using retrieval verification.
        
        Args:
            question: The question that was asked.
            answer: The answer to evaluate.
            top_k: Number of documents to retrieve.
            
        Returns:
            Dictionary with:
                - score: float, entailment score (0-1, higher = more supported)
                - is_hallucination: bool
                - details: dict, additional information
        """
        try:
            # Step 1: Retrieve evidence
            evidence = self.retrieve_evidence(question, top_k=top_k)
            
            # Step 2: Verify answer against evidence
            verification = self.verify_answer(answer, evidence)
            
            # Score is the max entailment score
            score = verification['entailment_score']
            
            # Threshold for hallucination detection
            threshold = 0.5
            is_hallucination = score < threshold
            
            return {
                'score': score,
                'is_hallucination': is_hallucination,
                'details': {
                    'entailment_score': verification['entailment_score'],
                    'contradiction_score': verification['contradiction_score'],
                    'n_evidence': len(evidence),
                    'best_evidence': verification.get('best_evidence'),
                    'threshold': threshold,
                    'signal_name': 'retrieval_verification'
                }
            }
        
        except Exception as e:
            logger.error(f"Retrieval verification failed: {e}")
            return {
                'score': 0.5,
                'is_hallucination': False,
                'details': {'error': str(e), 'signal_name': 'retrieval_verification'}
            }


# ============================================================
# Test Example
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Check if corpus exists
    corpus_dir = 'data/corpus'
    if not os.path.exists(os.path.join(corpus_dir, 'documents.pkl')):
        print("⚠️  Corpus not found. Run prepare_knowledge_corpus.py first.")
        print("   Running with limited functionality...")
    
    verifier = RetrievalVerifier(corpus_dir=corpus_dir)
    
    # Test 1: Factual answer
    print("\n" + "="*60)
    print("Test 1: Factual Answer")
    print("="*60)
    result = verifier.detect(
        question="What is the speed of light?",
        answer="The speed of light is approximately 299,792,458 meters per second."
    )
    print(f"Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    
    # Test 2: Hallucinated answer
    print("\n" + "="*60)
    print("Test 2: Hallucinated Answer")
    print("="*60)
    result = verifier.detect(
        question="What is the speed of light?",
        answer="The speed of light is exactly 500,000 kilometers per hour."
    )
    print(f"Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    
    # Test 3: Evidence retrieval
    print("\n" + "="*60)
    print("Test 3: Evidence Retrieval")
    print("="*60)
    evidence = verifier.retrieve_evidence("Who walked on the moon first?", top_k=3)
    for i, doc in enumerate(evidence):
        print(f"  {i+1}. [{doc['score']:.2f}] {doc['title']}: {doc['text'][:100]}...")
