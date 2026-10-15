import numpy as np
X = np.array([[0, 1], [1, 0]]); Z = np.diag([1, -1]); I = np.eye(2)
def k(ms):
    o = ms[0]
    for m in ms[1:]:
        o = np.kron(o, m)
    return o
psi = np.zeros(32)
for b in [0b11111, 0b10010, 0b01001, 0b10100, 0b01010, 0b00101, 0b11000, 0b01100]:
    psi[b] = 1
psi /= np.linalg.norm(psi)
S = {'ZZZZI': k([Z, Z, Z, Z, I]), 'ZIZZZ': k([Z, I, Z, Z, Z]), 'ZZIZZ': k([Z, Z, I, Z, Z]), 'IZZZZ': k([I, Z, Z, Z, Z])}
for n, s in S.items():
    print(n, '<S> =', round(float(psi @ s @ psi), 4))
# true [[5,1,3]] stabilizers XZZXI cyclic
T = [k([X, Z, Z, X, I]), k([I, X, Z, Z, X]), k([X, I, X, Z, Z]), k([Z, X, I, X, Z])]
print('true 5-qubit-code stabilizer expectations:', [round(float(psi @ t @ psi), 4) for t in T])
# Lindblad with L = Z1Z2 (unitary Hermitian): does it map error space to code space?
L = k([Z, Z, I]); e = np.zeros(8); e[0b101] = 1  # |101> = X on middle qubit of |111>
print('L|101> overlap with |111>:', float(np.zeros(8).tolist() and 0) if False else abs((L @ e)[7]))
# Kraus normalization in hybrid cells: K0 = (1-g dt/2) I, K1 = sqrt(g dt) L, L^2 = I
g, dt = 0.3, 0.05
print('sum K^dag K =', (1 - g*dt/2)**2 + g*dt, '(should be 1)')
