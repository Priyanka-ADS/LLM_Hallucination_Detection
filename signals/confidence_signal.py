"""
Confidence Detection Signal

Computes text-based confidence heuristics for hallucination detection.
Does NOT require model logits - works entirely on the text output.

Combines multiple heuristics:
1. Lexical uncertainty markers
2. Answer length analysis
3. Repetition detection
4. Specificity analysis  
5. Hedging language detection
"""

import re
import math
import logging
import numpy as np
from typing import Dict, List
from collections import Counter

logger = logging.getLogger(__name__)


class ConfidenceDetector:
    """
    Detects hallucinations using text-based confidence heuristics.
    
    The intuition: Hallucinated answers often exhibit telltale patterns:
    - Hedging language ("I think", "probably", "maybe")
    - Excessive verbosity or repetition
    - Lack of specific details (numbers, names, dates)
    - Unusual word distributions
    
    This detector does NOT require access to model logits or probabilities.
    """
    
    # Hedging/uncertainty phrases
    HEDGING_PHRASES = [
        'i think', 'i believe', 'probably', 'maybe', 'perhaps',
        'it seems', 'it appears', 'might be', 'could be', 'possibly',
        'not sure', 'not certain', 'unclear', 'approximately',
        'roughly', 'around', 'about', 'likely', 'unlikely',
        'as far as i know', 'to my knowledge', 'if i recall',
        'i\'m not sure', 'i don\'t know', 'hard to say',
        'it depends', 'some say', 'supposedly', 'allegedly',
        'reportedly', 'presumably', 'ostensibly'
    ]
    
    # Confident/assertive phrases
    CONFIDENT_PHRASES = [
        'is', 'are', 'was', 'were', 'definitely', 'certainly',
        'absolutely', 'clearly', 'obviously', 'undoubtedly',
        'in fact', 'specifically', 'exactly', 'precisely',
        'according to', 'studies show', 'research indicates'
    ]
    
    # Filler words that may indicate uncertainty
    FILLER_WORDS = [
        'basically', 'essentially', 'actually', 'literally',
        'kind of', 'sort of', 'like', 'well', 'um', 'uh',
        'you know', 'i mean'
    ]
    
    def __init__(self):
        """Initialize the ConfidenceDetector."""
        logger.info("ConfidenceDetector initialized")
    
    def _compute_hedging_score(self, text: str) -> float:
        """
        Compute a hedging score based on uncertainty markers.
        
        Returns:
            Score between 0 and 1. Higher = more hedging = less confident.
        """
        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)
        
        # Count hedging phrases
        hedge_count = 0
        for phrase in self.HEDGING_PHRASES:
            hedge_count += text_lower.count(phrase)
        
        # Count confident phrases
        confident_count = 0
        for phrase in self.CONFIDENT_PHRASES:
            confident_count += text_lower.count(phrase)
        
        # Count filler words
        filler_count = 0
        for phrase in self.FILLER_WORDS:
            filler_count += text_lower.count(phrase)
        
        # Normalize by word count
        hedge_ratio = (hedge_count + filler_count) / word_count
        confident_ratio = confident_count / word_count
        
        # Hedging score: more hedging + less confidence = higher score
        hedging_score = min(1.0, hedge_ratio * 10) - min(0.5, confident_ratio * 5)
        
        return max(0.0, min(1.0, hedging_score))
    
    def _compute_length_score(self, text: str) -> float:
        """
        Compute a confidence score based on answer length.
        
        Very short or very long answers may indicate issues.
        Medium-length, specific answers tend to be more reliable.
        
        Returns:
            Score between 0 and 1. Higher = more appropriate length = more confident.
        """
        word_count = len(text.split())
        
        # Optimal range: 5-50 words for a typical answer
        if word_count < 3:
            # Very short - possibly evasive
            return 0.3
        elif word_count < 5:
            return 0.5
        elif word_count <= 50:
            # Good length range
            return 0.8
        elif word_count <= 100:
            # Getting long - might be padding
            return 0.6
        else:
            # Very long - possibly rambling/hallucinating
            return 0.4
    
    def _compute_repetition_score(self, text: str) -> float:
        """
        Detect repetition in the answer.
        
        Hallucinated text often contains repetitive phrases or circular logic.
        
        Returns:
            Score between 0 and 1. Higher = more repetition = less confident.
        """
        words = text.lower().split()
        if len(words) < 5:
            return 0.0  # Too short to meaningfully check repetition
        
        # Check word-level repetition
        word_counts = Counter(words)
        # Remove common stop words from repetition check
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on',
                      'at', 'to', 'for', 'of', 'and', 'or', 'but', 'with', 'by',
                      'from', 'it', 'this', 'that', 'be', 'as', 'not', 'has', 'had'}
        
        content_words = [w for w in words if w not in stop_words and len(w) > 2]
        if not content_words:
            return 0.0
        
        content_counts = Counter(content_words)
        
        # Calculate ratio of unique content words
        unique_ratio = len(content_counts) / max(len(content_words), 1)
        
        # Check for repeated phrases (bigrams and trigrams)
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        bigram_repetition = 1.0 - (len(set(bigrams)) / max(len(bigrams), 1))
        trigram_repetition = 1.0 - (len(set(trigrams)) / max(len(trigrams), 1))
        
        # Combined repetition score
        repetition_score = (
            (1.0 - unique_ratio) * 0.4 +  # Word-level repetition
            bigram_repetition * 0.3 +       # Bigram repetition
            trigram_repetition * 0.3          # Trigram repetition
        )
        
        return max(0.0, min(1.0, repetition_score))
    
    def _compute_specificity_score(self, text: str) -> float:
        """
        Measure how specific/detailed the answer is.
        
        Specific answers with dates, numbers, proper nouns tend to be more reliable.
        Vague, generic answers may indicate hallucination.
        
        Returns:
            Score between 0 and 1. Higher = more specific = more confident.
        """
        text_lower = text.lower()
        word_count = max(len(text.split()), 1)
        
        specificity_indicators = 0
        
        # Check for numbers/dates
        numbers = re.findall(r'\b\d+[\.,]?\d*\b', text)
        specificity_indicators += min(len(numbers) * 2, 6)
        
        # Check for years
        years = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', text)
        specificity_indicators += len(years) * 2
        
        # Check for proper nouns (capitalized words not at start of sentence)
        proper_nouns = re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+', text)
        specificity_indicators += min(len(proper_nouns), 5)
        
        # Check for specific units of measurement
        units = re.findall(r'\b(km|miles|meters|feet|kg|pounds|celsius|fahrenheit|percent|%)\b', 
                          text_lower)
        specificity_indicators += len(units) * 2
        
        # Normalize
        specificity_score = min(1.0, specificity_indicators / (word_count * 0.5))
        
        return specificity_score
    
    def _compute_entropy_score(self, text: str) -> float:
        """
        Compute character-level entropy of the text.
        
        Very low or very high entropy can indicate issues.
        Natural language typically has moderate entropy.
        
        Returns:
            Score between 0 and 1. Higher entropy indicates more randomness.
        """
        if not text:
            return 0.0
        
        # Character frequency distribution
        char_counts = Counter(text.lower())
        total_chars = sum(char_counts.values())
        
        # Shannon entropy
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                prob = count / total_chars
                entropy -= prob * math.log2(prob)
        
        # Normalize to [0, 1] range
        # English text typically has entropy around 4-5 bits per character
        max_entropy = math.log2(len(char_counts)) if len(char_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def detect(self, question: str, answer: str) -> dict:
        """
        Detect potential hallucination using confidence heuristics.
        
        Args:
            question: The question that was asked.
            answer: The answer to evaluate.
            
        Returns:
            Dictionary with:
                - score: float, confidence score (0-1, higher = more confident)
                - is_hallucination: bool
                - details: dict, individual heuristic scores
        """
        try:
            # Compute individual heuristic scores
            hedging = self._compute_hedging_score(answer)
            length = self._compute_length_score(answer)
            repetition = self._compute_repetition_score(answer)
            specificity = self._compute_specificity_score(answer)
            entropy = self._compute_entropy_score(answer)
            
            # Combine scores with weights
            # Higher confidence = less likely hallucinated
            weights = {
                'hedging': -0.25,      # More hedging → less confident
                'length': 0.15,        # Good length → more confident
                'repetition': -0.20,   # More repetition → less confident
                'specificity': 0.25,   # More specific → more confident
                'entropy': 0.15,       # Moderate entropy → more confident
            }
            
            # Compute weighted score
            raw_score = (
                (1.0 - hedging) * abs(weights['hedging']) +
                length * abs(weights['length']) +
                (1.0 - repetition) * abs(weights['repetition']) +
                specificity * abs(weights['specificity']) +
                entropy * abs(weights['entropy'])
            )
            
            # Normalize to [0, 1]
            total_weight = sum(abs(w) for w in weights.values())
            confidence_score = raw_score / total_weight
            confidence_score = max(0.0, min(1.0, confidence_score))
            
            # Threshold
            threshold = 0.5
            is_hallucination = confidence_score < threshold
            
            return {
                'score': confidence_score,
                'is_hallucination': is_hallucination,
                'details': {
                    'hedging_score': hedging,
                    'length_score': length,
                    'repetition_score': repetition,
                    'specificity_score': specificity,
                    'entropy_score': entropy,
                    'threshold': threshold,
                    'signal_name': 'confidence'
                }
            }
        
        except Exception as e:
            logger.error(f"Confidence detection failed: {e}")
            return {
                'score': 0.5,
                'is_hallucination': False,
                'details': {'error': str(e), 'signal_name': 'confidence'}
            }


# ============================================================
# Test Example
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    detector = ConfidenceDetector()
    
    # Test 1: Confident, specific answer
    print("\n" + "="*60)
    print("Test 1: Confident, Specific Answer")
    print("="*60)
    result = detector.detect(
        question="When was the Declaration of Independence signed?",
        answer="The Declaration of Independence was signed on July 4, 1776, in Philadelphia, Pennsylvania."
    )
    print(f"Confidence Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    for k, v in result['details'].items():
        if k not in ('signal_name', 'threshold'):
            print(f"  {k}: {v:.4f}")
    
    # Test 2: Uncertain, hedging answer
    print("\n" + "="*60)
    print("Test 2: Uncertain, Hedging Answer")
    print("="*60)
    result = detector.detect(
        question="When was the Declaration of Independence signed?",
        answer="I think it was maybe around the 1770s or so, probably in some city, I'm not sure which one exactly."
    )
    print(f"Confidence Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    for k, v in result['details'].items():
        if k not in ('signal_name', 'threshold'):
            print(f"  {k}: {v:.4f}")
    
    # Test 3: Repetitive answer
    print("\n" + "="*60)
    print("Test 3: Repetitive Answer")
    print("="*60)
    result = detector.detect(
        question="What is photosynthesis?",
        answer="Photosynthesis is a process. The process of photosynthesis is a process where plants do photosynthesis. This photosynthesis process involves photosynthesis."
    )
    print(f"Confidence Score: {result['score']:.4f}")
    print(f"Is Hallucination: {result['is_hallucination']}")
    for k, v in result['details'].items():
        if k not in ('signal_name', 'threshold'):
            print(f"  {k}: {v:.4f}")
