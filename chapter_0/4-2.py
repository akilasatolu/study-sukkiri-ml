data = [71, 67, 73, 61, 79, 59, 83, 87, 72, 79]
scores = []
for d in data:
    scores.append(d)

final_scores = []
for s in scores:
    final_scores.append(0.8 * s + 20)

avarage_score = sum(final_scores) / len(final_scores)
print(avarage_score)