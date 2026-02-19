"""
Hallucination Detector - Main Ensemble Classifier

Combines multiple detection signals into an ensemble classifier:
1. Self-Consistency Signal
2. Retrieval Verification Signal
3. Confidence Signal

Uses LogisticRegression as the meta-classifier to combine signal scores.
"""

import os
import pickle
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Ensemble hallucination detector that combines multiple signals.
    
    Pipeline:
    1. Extract features from each signal detector
    2. Scale features using StandardScaler
    3. Classify using LogisticRegression
    4. Return prediction with confidence
    """
    
    def __init__(self, corpus_dir: str = 'data/corpus', 
                 use_signals: Optional[List[str]] = None):
        """
        Initialize the HallucinationDetector.
        
        Args:
            corpus_dir: Directory containing the knowledge corpus.
            use_signals: List of signals to use. Options: 
                         ['self_consistency', 'retrieval', 'confidence'].
                         If None, uses all signals.
        """
        self.corpus_dir = corpus_dir
        self.use_signals = use_signals or ['self_consistency', 'retrieval', 'confidence']
        
        # Initialize signal detectors
        self._detectors = {}
        self._init_detectors()
        
        # Classifier components
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            C=1.0,
            solver='lbfgs'
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Feature names for interpretability
        self.feature_names = self._get_feature_names()
        
        logger.info(f"HallucinationDetector initialized with signals: {self.use_signals}")
    
    def _init_detectors(self):
        """Initialize individual signal detectors."""
        if 'self_consistency' in self.use_signals:
            try:
                from signals.self_consistency import SelfConsistencyDetector
                self._detectors['self_consistency'] = SelfConsistencyDetector()
                logger.info("✅ Self-consistency detector loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load self-consistency detector: {e}")
        
        if 'retrieval' in self.use_signals:
            try:
                from signals.retrieval_verification import RetrievalVerifier
                self._detectors['retrieval'] = RetrievalVerifier(
                    corpus_dir=self.corpus_dir
                )
                logger.info("✅ Retrieval verification detector loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load retrieval detector: {e}")
        
        if 'confidence' in self.use_signals:
            try:
                from signals.confidence_signal import ConfidenceDetector
                self._detectors['confidence'] = ConfidenceDetector()
                logger.info("✅ Confidence detector loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load confidence detector: {e}")
    
    def _get_feature_names(self) -> List[str]:
        """Get list of feature names based on active signals."""
        features = []
        if 'self_consistency' in self.use_signals:
            features.append('self_consistency_score')
        if 'retrieval' in self.use_signals:
            features.extend([
                'retrieval_entailment_score',
                'retrieval_contradiction_score'
            ])
        if 'confidence' in self.use_signals:
            features.extend([
                'confidence_score',
                'hedging_score',
                'repetition_score',
                'specificity_score'
            ])
        return features
    
    def extract_features(self, question: str, answer: str) -> np.ndarray:
        """
        Extract features from all signal detectors for a single QA pair.
        
        Args:
            question: The question.
            answer: The answer to evaluate.
            
        Returns:
            numpy array of features.
        """
        features = []
        
        # Self-Consistency features
        if 'self_consistency' in self.use_signals:
            if 'self_consistency' in self._detectors:
                result = self._detectors['self_consistency'].detect(question, answer)
                features.append(result['score'])
            else:
                features.append(0.5)  # Default neutral score
        
        # Retrieval Verification features
        if 'retrieval' in self.use_signals:
            if 'retrieval' in self._detectors:
                result = self._detectors['retrieval'].detect(question, answer)
                features.append(result['score'])
                features.append(result['details'].get('contradiction_score', 0.0))
            else:
                features.extend([0.5, 0.0])
        
        # Confidence features
        if 'confidence' in self.use_signals:
            if 'confidence' in self._detectors:
                result = self._detectors['confidence'].detect(question, answer)
                features.append(result['score'])
                details = result.get('details', {})
                features.append(details.get('hedging_score', 0.0))
                features.append(details.get('repetition_score', 0.0))
                features.append(details.get('specificity_score', 0.0))
            else:
                features.extend([0.5, 0.0, 0.0, 0.5])
        
        return np.array(features, dtype=np.float64)
    
    def extract_features_batch(self, questions: List[str], answers: List[str],
                                show_progress: bool = True) -> np.ndarray:
        """
        Extract features for a batch of QA pairs.
        
        Args:
            questions: List of questions.
            answers: List of answers.
            show_progress: Whether to show progress bar.
            
        Returns:
            numpy array of shape (n_samples, n_features).
        """
        from tqdm import tqdm
        
        n_samples = len(questions)
        features_list = []
        
        iterator = enumerate(zip(questions, answers))
        if show_progress:
            iterator = tqdm(iterator, total=n_samples, desc="Extracting features")
        
        for i, (q, a) in iterator:
            try:
                features = self.extract_features(q, a)
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Feature extraction failed for sample {i}: {e}")
                # Use neutral features as fallback
                features_list.append(np.ones(len(self.feature_names)) * 0.5)
        
        return np.array(features_list)
    
    def train(self, questions: List[str], answers: List[str], 
              labels: List[int], val_questions: Optional[List[str]] = None,
              val_answers: Optional[List[str]] = None,
              val_labels: Optional[List[int]] = None) -> Dict:
        """
        Train the ensemble detector.
        
        Args:
            questions: Training questions.
            answers: Training answers.
            labels: Training labels (1=hallucinated, 0=truthful).
            val_questions: Optional validation questions.
            val_answers: Optional validation answers.
            val_labels: Optional validation labels.
            
        Returns:
            Dictionary with training metrics.
        """
        logger.info(f"Training with {len(questions)} samples...")
        
        # Extract features
        logger.info("Extracting training features...")
        X_train = self.extract_features_batch(questions, answers)
        y_train = np.array(labels)
        
        # Scale features
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train classifier
        logger.info("Training classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Compute training metrics
        train_pred = self.classifier.predict(X_train_scaled)
        train_acc = accuracy_score(y_train, train_pred)
        train_f1 = f1_score(y_train, train_pred)
        
        metrics = {
            'train_accuracy': train_acc,
            'train_f1': train_f1,
            'n_train_samples': len(questions),
            'n_features': X_train.shape[1],
            'feature_names': self.feature_names,
        }
        
        # Validation metrics if provided
        if val_questions and val_answers and val_labels:
            logger.info("Extracting validation features...")
            X_val = self.extract_features_batch(val_questions, val_answers)
            X_val_scaled = self.scaler.transform(X_val)
            y_val = np.array(val_labels)
            
            val_pred = self.classifier.predict(X_val_scaled)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
            metrics['val_f1'] = f1_score(y_val, val_pred)
        
        # Feature importance (coefficients)
        if hasattr(self.classifier, 'coef_'):
            importance = dict(zip(self.feature_names, 
                                self.classifier.coef_[0].tolist()))
            metrics['feature_importance'] = importance
        
        logger.info(f"Training complete! Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        return metrics
    
    def predict(self, question: str, answer: str) -> Dict:
        """
        Predict whether an answer is hallucinated.
        
        Args:
            question: The question.
            answer: The answer to evaluate.
            
        Returns:
            Dictionary with prediction results.
        """
        if not self.is_trained:
            logger.warning("Detector not trained. Using heuristic-only prediction.")
            return self._heuristic_predict(question, answer)
        
        # Extract features
        features = self.extract_features(question, answer)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.classifier.predict(features_scaled)[0]
        probabilities = self.classifier.predict_proba(features_scaled)[0]
        
        # Get individual signal results
        signal_results = {}
        for signal_name, detector in self._detectors.items():
            signal_results[signal_name] = detector.detect(question, answer)
        
        return {
            'is_hallucination': bool(prediction),
            'confidence': float(max(probabilities)),
            'hallucination_probability': float(probabilities[1]) if len(probabilities) > 1 else float(prediction),
            'features': dict(zip(self.feature_names, features.tolist())),
            'signal_results': signal_results
        }
    
    def _heuristic_predict(self, question: str, answer: str) -> Dict:
        """Fallback prediction using simple averaging of signal scores."""
        scores = []
        signal_results = {}
        
        for signal_name, detector in self._detectors.items():
            result = detector.detect(question, answer)
            scores.append(result['score'])
            signal_results[signal_name] = result
        
        avg_score = np.mean(scores) if scores else 0.5
        is_hallucination = avg_score < 0.5
        
        return {
            'is_hallucination': is_hallucination,
            'confidence': abs(avg_score - 0.5) * 2,
            'hallucination_probability': 1.0 - avg_score,
            'features': {},
            'signal_results': signal_results
        }
    
    def predict_batch(self, questions: List[str], answers: List[str]) -> List[Dict]:
        """Predict for a batch of QA pairs."""
        results = []
        for q, a in zip(questions, answers):
            results.append(self.predict(q, a))
        return results
    
    def save(self, filepath: str = 'models/saved/hallucination_detector.pkl'):
        """Save the trained model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'use_signals': self.use_signals,
            'corpus_dir': self.corpus_dir,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = 'models/saved/hallucination_detector.pkl',
             corpus_dir: str = 'data/corpus') -> 'HallucinationDetector':
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        detector = cls(
            corpus_dir=model_data.get('corpus_dir', corpus_dir),
            use_signals=model_data.get('use_signals')
        )
        
        # Restore trained components
        detector.classifier = model_data['classifier']
        detector.scaler = model_data['scaler']
        detector.is_trained = model_data['is_trained']
        detector.feature_names = model_data['feature_names']
        
        logger.info(f"Model loaded from {filepath}")
        return detector


# ============================================================
# Test Example
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Testing HallucinationDetector (untrained heuristic mode)")
    print("="*60)
    
    # Create detector with only confidence signal (fastest, no model download needed)
    detector = HallucinationDetector(use_signals=['confidence'])
    
    # Test predictions
    test_cases = [
        ("What is the capital of France?", 
         "The capital of France is Paris."),
        ("What is the speed of light?", 
         "I think it might be around 500 km/h or something like that, maybe."),
        ("When did WWII end?", 
         "World War II ended in 1945 with the surrender of Germany and Japan."),
    ]
    
    for question, answer in test_cases:
        result = detector.predict(question, answer)
        status = "🔴 HALLUCINATION" if result['is_hallucination'] else "🟢 TRUTHFUL"
        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"→ {status} (confidence: {result['confidence']:.4f})")
