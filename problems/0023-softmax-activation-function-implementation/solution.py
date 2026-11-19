import math

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    max_value = max(scores)
    nominator_values = ([math.exp(i-max_value) for i in scores])
    denomintator_value = sum(nominator_values)
    softmax_values = [i / denomintator_value for i in nominator_values]
    return softmax_values