from tools import classify_intent_heuristic

tests = [
    ("hi", "greeting"),
    ("what is pricing?", "product_inquiry"),
    ("i want to sign up", "high_intent"),
]

def evaluate():
    correct = 0

    for text, expected in tests:
        pred = classify_intent_heuristic(text).value
        print(f"{text} → {pred} (expected: {expected})")

        if pred == expected:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(tests)}")

if __name__ == "__main__":
    evaluate()