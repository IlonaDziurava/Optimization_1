import numpy as np


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


def Simplex_method(C, A, b, eps):
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


def interior_point_algorithm(C, A, x, b, epsilon, alpha, max_iterations=1000):
    try:
        # Check for valid initial x values (must be positive if required by the method)
        if np.any(x <= 0):
            return "The method is not applicable! \n", None

        for iteration in range(max_iterations):
            v = x
            D = np.diag(x)
            A_bar = np.dot(A, D)
            c_bar = np.dot(D, C)
            I = np.eye(len(C))

            # Compute F and check if it's invertible
            F = np.dot(A_bar, np.transpose(A_bar))
            if np.linalg.det(F) == 0:
                return "The method is not applicable! \n", None

            FI = np.linalg.inv(F)
            H = np.dot(np.transpose(A_bar), FI)
            P = np.subtract(I, np.dot(H, A_bar))
            c_p = np.dot(P, c_bar)

            nu = np.absolute(np.min(c_p))
            if nu == 0:
                return "The problem does not have a solution! \n", None

            y = np.add(np.ones(len(C), float), (alpha / nu) * c_p)
            y_bar = np.dot(D, y)
            x = y_bar

            if np.linalg.norm(np.subtract(y_bar, v), ord=2) < epsilon:
                optimum_value = np.dot(C, x)
                return x, optimum_value

        return "The problem does not have a solution! \n", None

    except np.linalg.LinAlgError:
        return "The method is not applicable! \n", None


def run_test_case(C, A, b, eps, case_num):
    print(f"\nTest Case {case_num }:")
    solver_state, solution, optimum_value = Simplex_method(C, A, b, eps)

    if solver_state == "solved":
        solution = [round(float(x), 2) for x in solution]
        print('Solution:', solution)
        print('Optimal value:', round(optimum_value, 2), "\n")
    elif solver_state == "unbounded":
        print('Unbounded \n')
    else:
        print('Error:', optimum_value)


test_case_1 = ([4, 5, 4], [[2, 3, 6], [4, 2, 4], [4, 6, 8]], [240, 200, 160], 0.0001)

run_test_case(*test_case_1, case_num=1)

# Example usage
C1 = np.array([4, 5, 4, 0, 0, 0], float)  # Objective function coefficients
A1 = np.array([[2, 3, 6, 1, 0, 0], [4, 2, 4, 0, 1, 0], [4, 6, 8, 0, 0, 1]], float)  # Constraint matrix
x_initial1 = np.array([1, 1, 1, 229, 190, 142], float)  # Initial starting point
b1 = np.array([240, 200, 160], float)  # Right-hand side numbers
epsilon1 = 0.0001  # Approximation accuracy

# Run the algorithm with α = 0.5
alpha_05 = 0.5
result_05_1, optimum_value_05_1 = interior_point_algorithm(C1, A1, x_initial1, b1, epsilon1, alpha_05)
print("Solution with α = 0.5:", result_05_1)
print('Optimal value:', round(optimum_value_05_1, 2), "\n")

# Run the algorithm with α = 0.9
alpha_09 = 0.9
result_09_1, optimum_value_09_1 = interior_point_algorithm(C1, A1, x_initial1, b1, epsilon1, alpha_09)
print("Solution with α = 0.9:", result_09_1)
print('Optimal value:', round(optimum_value_09_1, 2), "\n")

test_case_2 = ([10, 12, 8], [[3, 4, 2], [4, 3, 3], [5, 3, 5]], [1020, 940, 1010], 0.0001)

run_test_case(*test_case_2, case_num=2)

C2 = np.array([10, 12, 8, 0, 0, 0], float)  # Objective function coefficients
A2 = np.array([[3, 4, 2, 1, 0, 0], [4, 3, 3, 0, 1, 0], [5, 3, 5, 0, 0, 1]], float)  # Constraint matrix
x_initial2 = np.array([1, 1, 1, 1011, 930, 997], float)  # Initial starting point
b2 = np.array([1020, 940, 1010], float)  # Right-hand side numbers
epsilon2 = 0.0001  # Approximation accuracy

# Run the algorithm with α = 0.5
alpha_05 = 0.5
result_05_2, optimum_value_05_2 = interior_point_algorithm(C2, A2, x_initial2, b2, epsilon2, alpha_05)
print("Solution with α = 0.5:", result_05_2)
print('Optimal value:', round(optimum_value_05_2, 2), "\n")

# Run the algorithm with α = 0.9rwx
alpha_09 = 0.9
result_09_2, optimum_value_09_2 = interior_point_algorithm(C2, A2, x_initial2, b2, epsilon2, alpha_09)
print("Solution with α = 0.9:", result_09_2)
print('Optimal value:', round(optimum_value_09_2, 2), "\n")

test_case_3 = ([3, 4], [[1, 2], [1, 1], [2, 1]], [4, 3, 8], 0.0001)

run_test_case(*test_case_3, case_num=3)

C3 = np.array([3, 4, 0, 0, 0], float)  # Objective function coefficients
A3 = np.array([[1, 2, 1, 0, 0], [1, 1, 0, 1, 0], [2, 1, 0, 0, 1]], float)  # Constraint matrix
x_initial3 = np.array([1, 1, 1, 1, 5], float)  # Initial starting point
b3 = np.array([4, 3, 8], float)  # Right-hand side numbers
epsilon3 = 0.0001  # Approximation accuracy

# Run the algorithm with α = 0.5
alpha_05 = 0.5
result_05_3, optimum_value_05_3 = interior_point_algorithm(C3, A3, x_initial3, b3, epsilon3, alpha_05)
print("Solution with α = 0.5:", result_05_3)
print('Optimal value:', round(optimum_value_05_3, 2), "\n")

alpha_09 = 0.9
result_09_3, optimum_value_09_3 = interior_point_algorithm(C3, A3, x_initial3, b3, epsilon3, alpha_09)
print("Solution with α = 0.9:", result_09_3)
print('Optimal value:', round(optimum_value_09_3, 2), "\n")

test_case_4 = ([9, 10, 16], [[18, 15, 12], [6, 4, 8], [5, 3, 3]], [360, 192, 180], 0.0001)

run_test_case(*test_case_4, case_num=4)

C4 = np.array([9, 10, 16, 0, 0, 0], float)
A4 = np.array([[18, 15, 12, 1, 0, 0], [6, 4, 8, 0, 1, 0], [5, 3, 3, 0, 0, 1]], float)
x_initial4 = np.array([1, 1, 1, 315, 174, 169], float)
b4 = np.array([360, 192, 180], float)
epsilon4 = 0.0001

alpha_05 = 0.5
result_05_4, optimum_value_05_4 = interior_point_algorithm(C4, A4, x_initial4, b4, epsilon4, alpha_05)
print("Solution with α = 0.5:", result_05_4)
print('Optimal value:', round(optimum_value_05_4, 2), "\n")

alpha_09 = 0.9
result_09_4, optimum_value_09_4 = interior_point_algorithm(C4, A4, x_initial4, b4, epsilon4, alpha_09)
print("Solution with α = 0.9:", result_09_4)
print('Optimal value:', round(optimum_value_09_4, 2), "\n")

test_case_5 = ([3, 4], [[1, -2], [-1, -1], [0, -3]], [4, -3, -8], 0.0001)

run_test_case(*test_case_5, case_num=5)

C5 = np.array([3, -4], float)
A5 = np.array([[1, -2, 1, 0, 0], [-1, -1, 0, 1, 0], [0, 3, 0, 0, 1]], float)
x_initial5 = np.array([4, -2, 4, 1, -14], float)
b5 = np.array([-4, 3, 8], float)
epsilon5 = 0.0001

alpha_05 = 0.5
result_05_5, optimum_value_05_5 = interior_point_algorithm(C5, A5, x_initial5, b5, epsilon5, alpha_05)
print("Solution with α = 0.5:", result_05_5, "\n")

alpha_09 = 0.9
result_09_5, optimum_value_09_5 = interior_point_algorithm(C5, A5, x_initial5, b5, epsilon5, alpha_09)
print("Solution with α = 0.9:", result_09_5, "\n")

try:
    print("Write this problem in standard form \n")
    C_input = np.array(list(map(float, input("Enter values for C (space-separated): ").split())), float)
    A_input = np.array([list(map(float, input(f"Enter row {i+1} of A (space-separated): ").split())) for i in range(len(C_input))], float)
    x_initial_input = np.array(list(map(float, input("Enter initial x values (space-separated): ").split())), float)
    b_input = np.array(list(map(float, input("Enter values for b (space-separated): ").split())), float)
    epsilon_input = float(input("Enter the convergence threshold ε: "))
    alpha_05 = 0.5

    test_case = (C_input, [A_input], b_input, epsilon_input)
    run_test_case(*test_case, case_num=6)

    result_custom, optimum_value_custom = interior_point_algorithm(C_input, A_input, x_initial_input, b_input,
                                                                   epsilon_input, alpha_05)
    print("Custom input solution:", result_custom, "\n")
    print("Optimum value:", optimum_value_custom, "\n")

except ValueError:
    print("Invalid input, please ensure all values are numbers and formatted correctly.")
