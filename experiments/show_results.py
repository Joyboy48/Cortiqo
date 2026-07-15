import json

with open("results/baseline_results.json") as f:
    data = json.load(f)

m = data["metrics"]
print("=== BASELINE RESULTS ===")
print(f"Total questions : {m['n']}")
print(f"Exact Match     : {m['exact_match']['count']}/{m['n']} ({m['exact_match']['pct']}%)")
print(f"Avg F1          : {m['f1_avg']}")
print(f"Hop-1 Recall @5 : {m['hop1_recall']['count']}/{m['n']} ({m['hop1_recall']['pct']}%)")
print(f"Hop-2 Recall @5 : {m['hop2_recall']['count']}/{m['n']} ({m['hop2_recall']['pct']}%)")
print()

results = data["results"]

print("=== 5 CORRECT ANSWERS ===")
correct = [r for r in results if r["em"] == 1][:5]
for r in correct:
    print(f"Q: {r['question'][:70]}")
    print(f"   Gold: {r['answer']}  |  AI: {r['prediction']}")
    print()

print("=== 5 WRONG ANSWERS ===")
wrong = [r for r in results if r["em"] == 0][:5]
for r in wrong:
    print(f"Q: {r['question'][:70]}")
    print(f"   Gold: {r['answer']}")
    print(f"   AI  : {r['prediction']}")
    print(f"   Hop1 retrieved: {r['hop1_recalled']}  |  Hop2 retrieved: {r['hop2_recalled']}")
    print()
