import numpy as np
import pandas as pd
import sys


class Transportation:

    def __init__(self, cost, supply, demand):
        self.n, self.m = cost.shape
        self.table = np.zeros((self.n + 2, self.m + 2), dtype=object)
        self.table[1:-1, 1:-1] = cost.copy()
        self.table[-1, 1:-1] = demand.copy()
        self.table[1:-1, -1] = supply.copy()
        self.table[0, 1::] = [f"C{i}" for i in range(self.m)] + ['Supply']
        self.table[1::, 0] = [f"R{i}" for i in range(self.n)] + ['Demand']

    def setup_table(self, minimize=True):
        if not minimize:
            cost = self.table[1:-1, 1:-1]
            self.table[1:-1, 1:-1] = np.max(cost) - cost
        self.table[-1, -1] = self.table[1:-1, -1].sum()
        self.table = np.array(self.table, dtype=object)

    def print_frame(self, table):
        df = pd.DataFrame(table[1:, 1:])
        df.columns = table[0, 1:]
        df.index = table[1:, 0]
        print(df, '\n')

    def print_table(self, allocation):
        alloc = [[i, j] for i, j, _ in allocation]
        cost, total = [], 0
        for i, x in enumerate(self.table[1:-1, 0]):
            temp = []
            for j, y in enumerate(self.table[0, 1:-1]):
                v = self.table[i + 1, j + 1]
                try:
                    z = alloc.index([x, y])
                    cell = f"{v}({allocation[z, -1]})"
                    total += v * allocation[z, -1]
                except ValueError:
                    cell = f"{v}"
                temp.append(cell)
            cost.append(temp)
        table = self.table.copy()
        table[1:-1, 1:-1] = cost
        self.print_frame(np.array(table))
        print("TOTAL COST: {}".format(total))


class NorthWestCorner:
    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])
        if self.table[x, -1] < self.table[-1, y]:
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins
        elif self.table[x, -1] > self.table[-1, y]:
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins
        else:
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def solve(self, show_iter=False):
        while self.table.shape != (2, 2):
            x, y = 0, 0
            self.allocate(x + 1, y + 1)
            if show_iter:
                self.trans.print_frame(self.table)
        return np.array(self.alloc, dtype=object)


class VogelsApproximationMethod:
    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])
        if self.table[x, -1] < self.table[-1, y]:
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins
        elif self.table[x, -1] > self.table[-1, y]:
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins
        else:
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def penalty(self, cost):
        gaps = np.zeros(cost.shape[0])
        for i, c in enumerate(cost):
            try:
                x, y = sorted(c)[:2]
            except ValueError:
                x, y = c[0], 0
            gaps[i] = abs(x - y)
        return gaps

    def solve(self, show_iter=False):
        while self.table.shape != (2, 2):
            cost = self.table[1:-1, 1:-1]
            supply = self.table[1:-1, -1]
            demand = self.table[-1, 1:-1]
            n = cost.shape[0]
            row_penalty = self.penalty(cost)
            col_penalty = self.penalty(cost.T)
            P = np.append(row_penalty, col_penalty)
            max_alloc = -np.inf
            for i in np.where(P == max(P))[0]:
                if i - n < 0:
                    r = i
                    L = cost[r]
                else:
                    c = i - n
                    L = cost[:, c]
                for j in np.where(L == min(L))[0]:
                    if i - n < 0:
                        c = j
                    else:
                        r = j
                    alloc = min([supply[r], demand[c]])
                    if alloc > max_alloc:
                        max_alloc = alloc
                        x, y = r, c
            self.allocate(x + 1, y + 1)
            if show_iter:
                self.trans.print_frame(self.table)
        return np.array(self.alloc, dtype=object)


class RussellsApproximationMethod:
    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def allocate(self, x, y):
        mins = min([self.table[x, -1], self.table[-1, y]])
        self.alloc.append([self.table[x, 0], self.table[0, y], mins])
        if self.table[x, -1] < self.table[-1, y]:
            self.table = np.delete(self.table, x, 0)
            self.table[-1, y] -= mins
        elif self.table[x, -1] > self.table[-1, y]:
            self.table = np.delete(self.table, y, 1)
            self.table[x, -1] -= mins
        else:
            self.table = np.delete(self.table, x, 0)
            self.table = np.delete(self.table, y, 1)

    def solve(self, show_iter=False):
        while self.table.shape != (2, 2):
            cost = self.table[1:-1, 1:-1]
            n, m = cost.shape
            U = np.max(cost, 1)
            V = np.max(cost, 0)
            for i in range(n):
                for j in range(m):
                    self.table[i + 1, j + 1] -= U[i] + V[j]
            mins = np.min(self.table[1:-1, 1:-1])
            x, y = np.argwhere(self.table[1:-1, 1:-1] == mins)[0]
            self.allocate(x + 1, y + 1)
            if show_iter:
                self.trans.print_frame(self.table)
        return np.array(self.alloc, dtype=object)


class InitialTable:
    def __init__(self, trans):
        self.trans = trans
        self.table = trans.table.copy()
        self.alloc = []

    def draw(self):
        self.trans.print_frame(self.table)
        return np.array(self.alloc, dtype=object)


if __name__ == "__main__":

    # Prompt user for input for the supply array
    S = np.array(list(map(int, input("Enter the supply array: ").split())))
    if len(S) != 3:  # Ensure 3 sources
        print("The method is not applicable!")
        sys.exit()

    print("Enter the cost matrix as space-separated rows (3 rows, 4 columns):")
    C = []
    for i in range(3):  # Ensure 3 rows
        row = list(map(int, input(f"Row {i + 1}: ").split()))
        if len(row) != 4:  # Ensure 4 columns per row
            print("The method is not applicable!")
            sys.exit()
        C.append(row)
    C = np.array(C)

    # Prompt user for input for the demand array
    D = np.array(list(map(int, input("Enter the demand array: ").split())))
    if len(D) != 4:  # Ensure 4 destinations
        print("The method is not applicable!")
        sys.exit()

    if np.any(C < 0) or np.any(S < 0) or np.any(D < 0):
        print("The method is not applicable!")
        sys.exit()

    if np.sum(S) != np.sum(D):
        print("The problem is not balanced!")
        sys.exit()

    trans = Transportation(C, S, D)
    trans.setup_table(minimize=True)

    print("\nInitial Cost Table")
    initial = InitialTable(trans)
    initial.draw()

    print("\nNorth-West corner method")
    NWC = NorthWestCorner(trans)
    allocation = NWC.solve(show_iter=False)
    trans.print_table(allocation)

    print("\nVogel’s approximation method")
    trans = Transportation(C, S, D)
    trans.setup_table(minimize=True)
    VAM = VogelsApproximationMethod(trans)
    allocation = VAM.solve(show_iter=False)
    trans.print_table(allocation)

    print("\nRussell’s approximation method")
    trans = Transportation(C, S, D)
    trans.setup_table(minimize=True)
    RAM = RussellsApproximationMethod(trans)
    allocation = RAM.solve(show_iter=False)
    trans.print_table(allocation)
