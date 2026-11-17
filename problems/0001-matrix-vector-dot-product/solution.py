def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
    # Return a list where each element is the dot product of a row of 'a' with 'b'.
    # If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a) == 0:
        return -1
    if len(a[0]) != len(b):
        return -1
    resp = []
    for v in a:
        cumsum = 0
        for i, j in zip(v, b):
            cumsum += i*j
        resp.append(cumsum)
    return resp