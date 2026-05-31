# test_grammar.py
import sys
from grammar_engine import process_sentence

def run_tests():
    tests = [
        # 1. Subjective Pronoun Case Correction
        {
            "input": ["me", "want", "eat"],
            "expected": "I want something to eat."
        },
        # 2. Copula "to be" auxiliary insertion
        {
            "input": ["they", "sick"],
            "expected": "They are sick."
        },
        {
            "input": ["we", "sick"],
            "expected": "We are sick."
        },
        {
            "input": ["doctor", "sick"],
            "expected": "Doctor is sick."
        },
        # 3. Desire Verb Infinitive building
        {
            "input": ["we", "want", "visit", "doctor"],
            "expected": "We want to visit the doctor."
        },
        # 4. Article/Preposition stitching
        {
            "input": ["you", "meet", "parents"],
            "expected": "You meet the parents."
        },
        {
            "input": ["we", "wait", "you"],
            "expected": "We wait for you."
        },
        {
            "input": ["me", "visit", "college"],
            "expected": "I visit to the college."
        },
        # 5. Fallback phrase templates (still backward compatible)
        {
            "input": ["emergency", "hospital"],
            "expected": "There is an emergency! Take me to the hospital."
        },
        {
            "input": ["sick", "medicine"],
            "expected": "I am feeling sick and I need medicine."
        },
        # 6. Dynamic POS Fallback heuristics for completely unrecognized new words
        # "learn_new_sign" should guess as VERB after "want", "car" should guess as NOUN
        {
            "input": ["me", "want", "learn_new_sign"],
            "expected": "I want to learn_new_sign."
        },
        {
            "input": ["me", "buy", "car"],
            "expected": "I buy the car." 
        }
    ]

    failed = 0
    print("=" * 60)
    print("  RUNNING GRAMMAR ENGINE INTEGRATION TESTS")
    print("=" * 60)

    for idx, t in enumerate(tests, 1):
        result = process_sentence(t["input"])
        passed = (result == t["expected"])
        status = "PASSED" if passed else "FAILED"
        print(f"Test {idx:02d}: {status}")
        print(f"  Input   : {t['input']}")
        print(f"  Expected: {t['expected']}")
        print(f"  Result  : {result}")
        print("-" * 60)
        if not passed:
            failed += 1

    print("=" * 60)
    if failed == 0:
        print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print(f"[FAILURE] {failed} TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
