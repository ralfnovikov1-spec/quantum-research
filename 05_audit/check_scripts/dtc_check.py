import numpy as np

def run(L, eps, h, J=1.0, n_cycles=30, seed=42, theta_mode='cell7'):
    dim = 2**L
    z = [np.array([1.0 if not ((n >> i) & 1) else -1.0 for n in range(dim)]) for i in range(L)]
    rng = np.random.RandomState(seed)
    fields = rng.uniform(-h, h, L)
    H1d = np.zeros(dim)
    for i in range(L-1):
        H1d += J * z[i] * z[i+1]
    for i in range(L):
        H1d += fields[i] * z[i]
    theta = np.pi * (1.0 - eps) if theta_mode == 'cell7' else np.pi - eps
    c, s = np.cos(theta/2), np.sin(theta/2)
    u1 = np.array([[c, -1j*s], [-1j*s, c]])
    U2 = u1
    for _ in range(L-1):
        U2 = np.kron(U2, u1)
    UF = np.diag(np.exp(-1j*H1d)) @ U2
    psi = np.zeros(dim, complex); psi[0] = 1
    mags = []
    for t in range(n_cycles):
        p = np.abs(psi)**2
        mags.append(sum(p @ z[i] for i in range(L)) / L)
        psi = UF @ psi
    return np.array(mags)

# Cell 7 phase map: rows h, cols eps
eps_grid = np.linspace(0, 0.4, 15); h_grid = np.linspace(0, 4, 15)
print('cell7 score (L=8,30 cycles) rows h=0,1.14,2.0,4.0; cols eps=0,0.114,0.2,0.286,0.4')
for h in (h_grid[0], h_grid[4], h_grid[7], h_grid[14]):
    row = []
    for e in (eps_grid[0], eps_grid[4], eps_grid[7], eps_grid[10], eps_grid[14]):
        m = run(8, e, h)[5:]
        par = np.array([(-1)**t for t in range(5, 30)])
        row.append(round(abs(np.mean(m*par)), 3))
    print(f'h={h:.2f}', row)
# Cell 6 control: same params but J=0 (no interactions), and J=1 h=0
for J, h in ((1.0, 1.0), (0.0, 1.0), (1.0, 0.0)):
    m = run(10, 0.1, h, J=J, n_cycles=50, theta_mode='cell6')
    print(f'cell6-type L=10 eps=0.1 J={J} h={h}: late amplitude', round(float(np.mean(np.abs(m[-5:]))), 3), 'alternating', bool(np.all(np.sign(m[-5:])[1:] != np.sign(m[-5:])[:-1])))
