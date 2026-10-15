import numpy as np
from scipy.sparse import csr_matrix, identity, kron
from itertools import combinations

# --- SYK Hermiticity check (book_4 cell 6) ---
I = identity(2, format='csr'); X = csr_matrix([[0, 1], [1, 0]]); Y = csr_matrix([[0, -1j], [1j, 0]]); Z = csr_matrix([[1, 0], [0, -1]])
def maj(idx, n):
    k = idx // 2
    ops = [Z]*k + ([Y] if idx % 2 else [X]) + [I]*(n-k-1)
    f = ops[0]
    for o in ops[1:]:
        f = kron(f, o, format='csr')
    return f
N = 10; n = 5; dim = 2**n
chis = [maj(i, n) for i in range(N)]
np.random.seed(42)
sd = np.sqrt(6/N**3)
H = csr_matrix((dim, dim), dtype=complex)
for (i, j, k, l) in combinations(range(N), 4):
    H += (1j*np.random.normal(0, sd)) * chis[i].dot(chis[j]).dot(chis[k]).dot(chis[l])
Hd = H.toarray()
print('SYK ||H - H^dag|| =', np.linalg.norm(Hd - Hd.conj().T), ' ||H + H^dag|| =', np.linalg.norm(Hd + Hd.conj().T), ' ||H||=', np.linalg.norm(Hd))

# --- Wormhole measurement side check (book_4 cell 5) ---
L = 6; dim = 2**L
sz, sx, sy = [], [], []
for i in range(L):
    d = np.array([1.0 if not ((m >> i) & 1) else -1.0 for m in range(dim)])
    sz.append(np.diag(d))
    r = np.arange(dim); c = r ^ (1 << i)
    Xm = np.zeros((dim, dim)); Xm[r, c] = 1; sx.append(Xm)
    Ym = np.zeros((dim, dim), complex); Ym[r, c] = np.array([1j if not ((m >> i) & 1) else -1j for m in range(dim)]); sy.append(Ym)
Hs = sum(sz[i] @ sz[i+1] for i in range(L-1)) + sum(-1.05*sx[i] + 0.5*sz[i] for i in range(L))
ev, U = np.linalg.eigh(Hs)
w = np.exp(-0.5*ev/2); w /= np.linalg.norm(w)
zv = np.array([[1 if ((m >> i) & 1) == 0 else -1 for i in range(L)] for m in range(dim)])
def proto(t, g, right_evolve=True):
    Uf = np.diag(np.exp(-1j*ev*t)); Ub = np.diag(np.exp(1j*ev*t))
    VE = U.T @ sy[0] @ U
    M = U @ (Uf @ VE @ Ub @ np.diag(w)) @ U.T
    M = M * np.exp(1j*g*(zv @ zv.T))
    if right_evolve:
        M = M @ (U @ Uf @ U.T)
    nrm = np.trace(M @ M.conj().T).real
    code_val = (np.trace(M @ M.conj().T @ sy[0].T)/nrm).real
    # true right-side expectation: <I x Y> = Tr(M^T M^* ... ) computed via full vector
    psi = M.reshape(-1)
    full_R = np.kron(np.eye(dim), sy[0]); full_L = np.kron(sy[0], np.eye(dim))
    R = (psi.conj() @ full_R @ psi).real/nrm; Lv = (psi.conj() @ full_L @ psi).real/nrm
    return code_val, Lv, R
for t, g in [(5.0, 2.0), (1.72, 0.69)]:
    a = proto(t, g, True); b = proto(t, g, False)
    print(f't={t} g={g}: code={a[0]:.4f} trueLeft<Y_L>={a[1]:.4f} trueRight<Y_R>={a[2]:.4f} | code without right evolution={b[0]:.4f}')
