def min_edit_dist(source, target):
    INS_COST = 1
    DEL_COST = 1

    n = len(source)
    m = len(target)
    # Distance matrix
    D = [ [0 for _ in range(m + 1)] for _ in range(n + 1)]
    
    # Initialization: zeroth row and column is the distance from the empty string
    a = [0] * (n + 1)
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + DEL_COST
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + INS_COST

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j] + DEL_COST,
                D[i - 1][j - 1] + sub_cost(source[i - 1], target[j - 1]),
                D[i][j - 1] + INS_COST,
            )
    print(D)

    return D[n][m]

def sub_cost(a, b):
    return (0 if a == b else 2)

if __name__ == "__main__":
    print(min_edit_dist("intention", "execution"))
