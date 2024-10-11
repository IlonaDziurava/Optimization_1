import numpy as np

# Default epsilon value
eps_default = 1e-6

def to_tableau(c, A, b):
    xb = [eq + [x] for eq, x in zip(A, b)]
    z = c + [0]
    return xb + [z]

def can_be_improved(tableau, epsilon):
    z = tableau[-1]
    return any(x > epsilon for x in z[:-1])

def get_pivot_position(tableau):
    z = tableau[-1]

    if all(x <= 0 for x in z[:-1]):
        raise ValueError("The method is not applicable!")

    column = next(i for i, x in enumerate(z[:-1]) if x > 0)

    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        if el <= 0:
            restrictions.append(np.inf)
        else:
            restrictions.append(eq[-1] / el)

    if all(r == np.inf for r in restrictions):
        raise ValueError("The method is not applicable!")

    row = restrictions.index(min(restrictions))
    return row, column

def pivot_step(tableau, pivot_position):
    new_tableau = [[] for eq in tableau]

    i, j = pivot_position
    pivot_value = tableau[i][j]
    
    new_tableau[i] = np.array(tableau[i]) / pivot_value
    
    for eq_i in range(len(tableau)):
        if eq_i != i:
            multiplier = np.array(new_tableau[i]) * tableau[eq_i][j]
            new_tableau[eq_i] = np.array(tableau[eq_i]) - multiplier

    return new_tableau

def is_basic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

def get_solution(tableau):
    columns = np.array(tableau).T
    num_vars = len(columns) - 1
    solutions = [0] * num_vars
    
    for column in columns[:-1]:
        if is_basic(column):
            one_index = column.tolist().index(1)
            solutions[one_index] = columns[-1][one_index]
            
    return solutions

def print_optimization_problem(C, A, b):
    n = len(C)
    m = len(A)
    
    print(f"max z = {' + '.join([f'{C[i]} * x{i+1}' for i in range(n)])}")
    print("subject to the constraints:")
    for i in range(m):
        print(f"{' + '.join([f'{A[i][j]} * x{j+1}' for j in range(n)])} <= {b[i]}")

def Simplex_method(C, A, b, eps=eps_default):
    print_optimization_problem(C, A, b)
    
    tableau = to_tableau(C, A, b)

    try:
        while can_be_improved(tableau, eps):
            pivot_position = get_pivot_position(tableau)
            tableau = pivot_step(tableau, pivot_position)
        
        solution = get_solution(tableau)
        optimum_value = -tableau[-1][-1]
        return "solved", solution, optimum_value

    except ValueError as e:
        return "unbounded", None, str(e)

# Test cases
test_cases = [
    ([1, 1, 0, 0, 0], [[-1, 1, 1, 0, 0], [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]], [2, 4, 4], 1e-6),
    ([4, 5, 4], [[2, 3, 6], [4, 2, 4], [4, 6, 8]], [240, 200, 160], 1e-6),
    ([3, 4], [[1, 2], [1, 1], [2, 1]], [4, 3, 8], 1e-4),
    ([10, 12, 8], [[3, 4, 2], [4, 3, 3], [5, 3, 5]], [1020, 940, 1010], 1e-4),
    ([3, 4], [[1, -2], [-1, -1], [0, -3]], [4, -3, -8], 1e-9)
]

for i, (C, A, b, eps) in enumerate(test_cases):
    print(f"\nTest Case {i+1}:")
    solver_state, solution, optimum_value = Simplex_method(C, A, b, eps)
    
    if solver_state == "solved":
        solution = [round(float(x), 2) for x in solution]
        print('solution:', solution)
        print('optimal value:', round(optimum_value, 2))
    else:
        print(optimum_value)
