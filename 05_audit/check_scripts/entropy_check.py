import numpy as np
from scipy.integrate import solve_ivp
from collections import Counter

def lorenz(t, s, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = s
    return [sigma*(y-x), x*(rho-z)-y, x*y-beta*z]

dt = 0.01
sol = solve_ivp(lorenz, (0, 200), [1, 1, 1], t_eval=np.arange(0, 200, dt), rtol=1e-9, atol=1e-9)
x, y, z = sol.y
b = int(20/dt); x, y, z = x[b:], y[b:], z[b:]
s = ((x >= 0).astype(int) << 2) | ((y >= 0).astype(int) << 1) | (z >= 24).astype(int)

def kl(seq, L, alpha):
    blocks = [tuple(seq[i:i+L]) for i in range(len(seq)-L+1)]
    c = Counter(blocks); N = len(blocks); K = len(c)
    tot = 0.0; unseen_mass = 0.0
    for w, n in c.items():
        pf = (n+alpha)/(N+alpha*K); nr = c.get(w[::-1], 0); pr = (nr+alpha)/(N+alpha*K)
        tot += pf*np.log(pf/pr)
        if nr == 0: unseen_mass += pf
    return tot, unseen_mass

for L in range(2, 7):
    r = [kl(s, L, a)[0] for a in (1e-10, 1e-3, 0.5)]
    print(L, 'KL alpha=1e-10,1e-3,0.5:', [round(v, 4) for v in r], 'mass of words whose reverse never seen:', round(kl(s, L, 1e-10)[1], 4))

H = []
for L in range(1, 11):
    c = Counter(tuple(s[i:i+L]) for i in range(len(s)-L+1)); p = np.array(list(c.values()))/sum(c.values())
    H.append(-(p*np.log(p)).sum())
print('H(L+1)-H(L):', np.round(np.diff(H), 4))
# palindromic share for L=4
c = Counter(tuple(s[i:i+4]) for i in range(len(s)-3)); N = sum(c.values())
print('fraction of L=4 windows that are palindromes (sigma=0):', round(sum(n for w, n in c.items() if w == w[::-1])/N, 4))
