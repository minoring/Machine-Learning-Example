from enum import Enum


def min_edit_dist(source, target):
    n = len(source)
    m = len(target)
    # Distance matrix
    D = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    ptr = [[0 for _ in range(m + 1)] for _ in range(n + 1)]
    # Initialization: zeroth row and column is the distance from the empty string
    a = [0] * (n + 1)
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + DEL_COST
        ptr[i][0] = DOWN
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + INS_COST
        ptr[0][j] = LEFT

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j] + DEL_COST,
                D[i][j - 1] + INS_COST,
                D[i - 1][j - 1] + sub_cost(source[i - 1], target[j - 1]),
            )
            ptr[i][j] = (
                DIAG
                if D[i - 1][j - 1] + sub_cost(source[i - 1], target[j - 1]) == D[i][j]
                else DOWN
                if D[i - 1][j] + DEL_COST < D[i][j - 1] + INS_COST
                else LEFT
            )
    print(back_trace(ptr, source, target))

    return D[n][m]


def back_trace(ptr, source, target):
    i = len(source)
    j = len(target)

    source_align = ""
    target_align = ""

    while i >= 1 and j >= 1:
        print("Compare i : ", i, "j : ", j)
        if ptr[i][j] == DIAG:
            source_align = source[i - 1] + source_align
            target_align = target[j - 1] + target_align
            i -= 1
            j -= 1
        elif ptr[i][j] == DOWN:
            source_align = source[i - 1] + source_align
            target_align = "*" + target_align
            i -= 1
        else:
            source_align = "*" + source_align
            target_align = target[j - 1] + target_align
            j -= 1

    while i >= 1:
        if ptr[i][j] == DOWN:
            source_align = source[i - 1] + source_align
            target_align = "*" + target_align
        i -= 1
    while j >= 1:
        if ptr[i][j] == LEFT:
            source_align = "*" + source_align
            target_align = target[j - 1] + target_align
        j -= 1

    return (source_align, target_align)


def sub_cost(a, b):
    return 0 if a == b else 2


if __name__ == "__main__":
    INS_COST = 1
    DEL_COST = 1
    DOWN = 0
    LEFT = 1
    DIAG = 2
    print(min_edit_dist("intention", "execution"))
