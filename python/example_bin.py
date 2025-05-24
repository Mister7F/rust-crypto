from rust_crypto import ExpressionBin, ExpressionBinConfig, MatrixBin
from sage import all_cmdline as sage
import random
import time
from utils import assert_raises

print("Start benchmark...")

config = ExpressionBinConfig()
a = config.gen("a")
b = config.gen("b")


assert str(a) == "a"
assert str(a ^ b) == "a + b"
assert str(a + b + 1 + a) == "1 + b"

assert bool(a)
assert bool(a * 1)
assert not bool(a * 0)
assert not bool(a ^ a)
assert bool(a ^ a ^ 1)


m, t = ExpressionBin.to_matrix([a ^ b, a ^ b ^ 1])
with assert_raises(ValueError, "Impossible system"):
    m.solve_right(t)

m, t = ExpressionBin.to_matrix([a ^ b, a ^ 1])
assert m.solve_right(t) == [True, True]

# Add a lot of dummy variables
cs = [config.gen(f"c{i}") for i in range(230)]
m, t = ExpressionBin.to_matrix([a ^ b, a ^ 1, cs[9] ^ 1])
assert m.solve_right(t) == [True, True] + [False] * 9 + [True] + [False] * 220

assert a.degree() == 1
assert (a + b + 1).degree() == 1
assert (a + b + 1 + a + b).degree() == 0

A = MatrixBin.from_list(
    [
        [1, 0, 1, 0],
        [1, 0, 1, 1],
    ],
)
B = MatrixBin.from_list(
    [
        [1, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ],
)
assert (A * B).to_list() == [[False, True], [False, False]]

###############
# Solve Right #
###############
print("\nsolve_right")

M = [
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
]
t = [False, True, True, False, True, True]
A = MatrixBin.from_list(M)
assert A.solve_right(t) == [False, False, False, True]

random.seed(133337)  # Make solution possible
M = [[random.randrange(2) for _ in range(2_001)] for _ in range(2_001)]
t = [random.randrange(2) for _ in range(2_001)]

M_s = sage.Matrix(sage.GF(2), M)
t_s = sage.vector(t)
start = time.time()
S_s = M_s.solve_right(t_s)
print("Sage     ", time.time() - start)

M_c = MatrixBin.from_list(M)
t_c = t
start = time.time()
S_c = M_c.solve_right(t_c)
print("MatrixBin", time.time() - start)
assert all(a == b for a, b in zip(S_s, S_c))

############
# Addition #
############
print("\nAddition")

random.seed(133337)
M1 = [[random.randrange(2) for _ in range(2_001)] for _ in range(2_001)]
M2 = [[random.randrange(2) for _ in range(2_001)] for _ in range(2_001)]

M1_s = sage.Matrix(sage.GF(2), M1)
M2_s = sage.Matrix(sage.GF(2), M2)
start = time.time()
S_s = M1_s + M2_s
print("Sage     ", time.time() - start)

M1_c = MatrixBin.from_list(M1)
M2_c = MatrixBin.from_list(M2)
t_c = t
start = time.time()
S_c = M1_c + M2_c
print("MatrixBin", time.time() - start)
assert all(
    a == b for line_a, line_b in zip(S_s, S_c.to_list()) for a, b in zip(line_a, line_b)
)

with assert_raises(ValueError, "Dimensions not compatible"):
    MatrixBin.from_list([[0, 0, 0], [0, 0, 0]]) + MatrixBin.from_list([[0, 0, 0]])

with assert_raises(ValueError, "Dimensions not compatible"):
    MatrixBin.from_list([[0, 0, 0], [0, 0, 0]]) + MatrixBin.from_list([[0], [0]])

###########
# Loading #
###########
print("\nLoading from python")

C = MatrixBin.from_list([[False, True], [True, False]])
M = sage.Matrix(sage.GF(2), [[False, True], [True, False]])
assert [list(map(int, line)) for line in M**-1] == C.inverse().to_list()

random.seed(133337)  # Make `M` invertible
M = [[random.randrange(2) for _ in range(2_001)] for _ in range(2_001)]

start = time.time()
M_s = sage.Matrix(sage.GF(2), M)
print("Sage     ", time.time() - start)

start = time.time()
M_c = MatrixBin.from_list(M)
print("MatrixBin", time.time() - start)

#############
# Inversion #
#############
print("\nInversion")
start = time.time()
I_s = M_s**-1
print("Sage     ", time.time() - start)

start = time.time()
I_c = M_c.inverse()
print("MatrixBin", time.time() - start)

assert [list(map(int, line)) for line in I_s] == I_c.to_list()


##################
# Multiplication #
##################
print("\nLoading from python big matrix")

A = [[random.randrange(2) for _ in range(10_001)] for _ in range(10_008)]
B = [[random.randrange(2) for _ in range(1_321)] for _ in range(10_001)]

start = time.time()
A_s = sage.Matrix(sage.GF(2), A)
B_s = sage.Matrix(sage.GF(2), B)
print("Sage     ", time.time() - start)

start = time.time()
A_c = MatrixBin.from_list(A)
B_c = MatrixBin.from_list(B)
print("MatrixBin", time.time() - start)

print("\nMultiplication")

start = time.time()
R_s = A_s * B_s
print("Sage     ", time.time() - start)

start = time.time()
R_c = A_c * B_c
print("MatrixBin", time.time() - start)

assert [list(map(int, line)) for line in R_s] == R_c.to_list()

with assert_raises(ValueError, "Dimensions not compatible"):
    MatrixBin.from_list([[0, 0, 0], [0, 0, 0]]) + MatrixBin.from_list([[0, 0, 0]])

###########
# Echelon #
###########
print("\nEchelon")
start = time.time()
E_s = A_s.echelon_form()
print("Sage     ", time.time() - start)

start = time.time()
E_c = A_c.echelon_form([False] * len(A))[0]
print("MatrixBin", time.time() - start)
