import math

def compare_softmax(scores: list) -> dict:
    """Compare naive and numerically stable softmax."""
    naive = naive_softmax(scores)
    stable = stable_softmax(scores)

    max_diff = 0
    for i, j in zip(naive, stable):
        if math.isnan(i):
            max_diff = float('nan')
            break
        diff = abs(i - j)
        max_diff = diff if diff > max_diff else max_diff

    return {'naive': [round(x,6) for x in naive] , 'stable': [round(x,6) for x in stable] , 'max_abs_diff': round(max_diff, 6)}

def naive_softmax(scores: list) -> list:
    try:
        nominator = [math.exp(i) for i in scores]
        denominator = sum(nominator)
        return [i / denominator for i in nominator]
    except OverflowError:
        return [float('nan') for _ in scores]

def stable_softmax(scores: list) -> list:
    max_score = max(scores)
    nominator = [math.exp(i - max_score) for i in scores]
    denominator = sum(nominator)
    return [i / denominator for i in nominator]