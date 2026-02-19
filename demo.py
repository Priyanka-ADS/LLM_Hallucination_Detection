#!/usr/bin/env python3
"""
Interactive Demo Script for Hallucination Detection

Provides a user-friendly demo that:
1. Loads the trained detector
2. Accepts question/answer pairs
3. Shows step-by-step detection process
4. Explains decisions in plain English
"""

import os
import sys
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class HallucinationDemo:
    """Interactive demonstration of the hallucination detector."""
    
    def __init__(self):
        """Initialize the demo."""
        self.detector = None
        self._load_detector()
    
    def _load_detector(self):
        """Load the trained detector."""
        model_path = 'models/saved/hallucination_detector.pkl'
        
        if os.path.exists(model_path):
            try:
                from detector import HallucinationDetector
                self.detector = HallucinationDetector.load(model_path)
                print("✅ Trained model loaded successfully!")
            except Exception as e:
                print(f"⚠️  Could not load trained model: {e}")
                print("   Using heuristic-only mode.")
                from detector import HallucinationDetector
                self.detector = HallucinationDetector(use_signals=['confidence'])
        else:
            print("⚠️  No trained model found. Using heuristic-only mode.")
            print("   Run train_detector.py first for better results.")
            from detector import HallucinationDetector
            self.detector = HallucinationDetector(use_signals=['confidence'])
    
    def analyze(self, question, answer):
        """
        Analyze a question-answer pair for hallucination.
        Shows the step-by-step detection process.
        """
        print("\n" + "="*70)
        print("🔍 HALLUCINATION DETECTION ANALYSIS")
        print("="*70)
        
        print(f"\n📝 Question: {question}")
        print(f"📝 Answer:   {answer}")
        
        # Get prediction
        result = self.detector.predict(question, answer)
        
        # Step 1: Show individual signal results
        print(f"\n{'─'*70}")
        print("📊 Step 1: Individual Signal Analysis")
        print(f"{'─'*70}")
        
        signal_results = result.get('signal_results', {})
        
        if 'self_consistency' in signal_results:
            sc = signal_results['self_consistency']
            score = sc['score']
            bar = self._score_bar(score)
            print(f"\n  🔄 Self-Consistency: {score:.4f} {bar}")
            print(f"     Measures if the answer is internally consistent")
            print(f"     {'✅ High consistency' if score > 0.5 else '⚠️  Low consistency - possible hallucination'}")
        
        if 'retrieval' in signal_results:
            rv = signal_results['retrieval']
            score = rv['score']
            bar = self._score_bar(score)
            print(f"\n  📚 Retrieval Verification: {score:.4f} {bar}")
            print(f"     Checks if evidence supports the answer")
            if rv.get('details', {}).get('best_evidence'):
                evidence = rv['details']['best_evidence']
                print(f"     Best evidence: {evidence.get('text', '')[:100]}...")
            print(f"     {'✅ Supported by evidence' if score > 0.5 else '⚠️  Not well supported - possible hallucination'}")
        
        if 'confidence' in signal_results:
            conf = signal_results['confidence']
            score = conf['score']
            bar = self._score_bar(score)
            print(f"\n  💯 Confidence Analysis: {score:.4f} {bar}")
            details = conf.get('details', {})
            if details:
                print(f"     Hedging:     {details.get('hedging_score', 0):.3f}")
                print(f"     Repetition:  {details.get('repetition_score', 0):.3f}")
                print(f"     Specificity: {details.get('specificity_score', 0):.3f}")
            print(f"     {'✅ Confident answer' if score > 0.5 else '⚠️  Low confidence - possible hallucination'}")
        
        # Step 2: Feature values
        print(f"\n{'─'*70}")
        print("📊 Step 2: Combined Features")
        print(f"{'─'*70}")
        
        features = result.get('features', {})
        if features:
            for name, value in features.items():
                print(f"  {name:<35} {value:.4f}")
        
        # Step 3: Final prediction
        print(f"\n{'─'*70}")
        print("🎯 Step 3: Final Prediction")
        print(f"{'─'*70}")
        
        is_hallucination = result['is_hallucination']
        confidence = result['confidence']
        prob = result.get('hallucination_probability', 0.5)
        
        if is_hallucination:
            print(f"\n  🔴 HALLUCINATION DETECTED")
            print(f"     Confidence: {confidence:.1%}")
            print(f"     Probability: {prob:.1%}")
            print(f"\n  💡 Explanation:")
            print(f"     The answer shows signs of being hallucinated.")
            self._explain_hallucination(signal_results)
        else:
            print(f"\n  🟢 ANSWER APPEARS TRUTHFUL")
            print(f"     Confidence: {confidence:.1%}")
            print(f"     Probability of hallucination: {prob:.1%}")
            print(f"\n  💡 Explanation:")
            print(f"     The answer appears to be factually grounded.")
            self._explain_truthful(signal_results)
        
        print(f"\n{'='*70}\n")
        return result
    
    def _score_bar(self, score, width=20):
        """Create a visual score bar."""
        filled = int(score * width)
        empty = width - filled
        if score > 0.7:
            color = "🟢"
        elif score > 0.4:
            color = "🟡"
        else:
            color = "🔴"
        return f"{color} [{'█' * filled}{'░' * empty}]"
    
    def _explain_hallucination(self, signal_results):
        """Explain why the answer was flagged as hallucinated."""
        reasons = []
        
        if 'self_consistency' in signal_results:
            if signal_results['self_consistency']['score'] < 0.5:
                reasons.append("- The answer shows low internal consistency")
        
        if 'retrieval' in signal_results:
            if signal_results['retrieval']['score'] < 0.5:
                reasons.append("- The answer is not well supported by available evidence")
        
        if 'confidence' in signal_results:
            details = signal_results['confidence'].get('details', {})
            if details.get('hedging_score', 0) > 0.3:
                reasons.append("- The answer contains hedging/uncertainty language")
            if details.get('repetition_score', 0) > 0.3:
                reasons.append("- The answer shows excessive repetition")
            if details.get('specificity_score', 0) < 0.3:
                reasons.append("- The answer lacks specific details")
        
        if reasons:
            for reason in reasons:
                print(f"     {reason}")
        else:
            print("     The combined signal scores suggest hallucination.")
    
    def _explain_truthful(self, signal_results):
        """Explain why the answer appears truthful."""
        reasons = []
        
        if 'self_consistency' in signal_results:
            if signal_results['self_consistency']['score'] > 0.5:
                reasons.append("- The answer is internally consistent")
        
        if 'retrieval' in signal_results:
            if signal_results['retrieval']['score'] > 0.5:
                reasons.append("- The answer is supported by available evidence")
        
        if 'confidence' in signal_results:
            details = signal_results['confidence'].get('details', {})
            if details.get('specificity_score', 0) > 0.3:
                reasons.append("- The answer contains specific details (numbers, names, dates)")
            if details.get('hedging_score', 0) < 0.2:
                reasons.append("- The answer uses confident, assertive language")
        
        if reasons:
            for reason in reasons:
                print(f"     {reason}")
        else:
            print("     The combined signal scores support truthfulness.")


def run_interactive_demo():
    """Run the interactive demo loop."""
    print("\n" + "="*70)
    print("🌟 HALLUCINATION DETECTOR - INTERACTIVE DEMO")
    print("="*70)
    print("\nThis tool analyzes question-answer pairs to detect potential")
    print("hallucinations in LLM outputs.\n")
    
    demo = HallucinationDemo()
    
    # Sample examples
    examples = [
        ("What is the capital of France?", 
         "The capital of France is Paris, located on the Seine River."),
        ("What is the speed of light?", 
         "The speed of light is about 500 km/h, it was discovered by Newton in 1850."),
        ("When was the Declaration of Independence signed?",
         "I think it was maybe around the 1700s or so, probably in some American city."),
    ]
    
    print("📋 Try these examples or enter your own:\n")
    for i, (q, a) in enumerate(examples, 1):
        print(f"  Example {i}:")
        print(f"    Q: {q}")
        print(f"    A: {a}\n")
    
    while True:
        print("─" * 70)
        print("Enter 'q' to quit, 'e1/e2/e3' for examples, or type your own:")
        
        user_input = input("\n📌 Question (or command): ").strip()
        
        if user_input.lower() == 'q':
            print("\n👋 Goodbye!")
            break
        
        if user_input.lower() in ('e1', 'e2', 'e3'):
            idx = int(user_input[-1]) - 1
            question, answer = examples[idx]
        else:
            question = user_input
            answer = input("📌 Answer: ").strip()
        
        if question and answer:
            demo.analyze(question, answer)


def main():
    """Main function."""
    if len(sys.argv) > 2:
        # Command line mode
        question = sys.argv[1]
        answer = sys.argv[2]
        demo = HallucinationDemo()
        demo.analyze(question, answer)
    else:
        # Interactive mode
        run_interactive_demo()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
