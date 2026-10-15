# Single-excitation re-simulation of crystal_quantum cell 16 (Thouless pump) and
# cell 26 premise (X noise on |+>). ISWAP**t acts on span{|01>,|10>} as
# [[cos j, i sin j],[i sin j, cos j]] with j = t*pi/2.
import numpy as np

def pump(n=8, cycles=4, spc=20, start=0):
    psi = np.zeros(n, complex); psi[start] = 1
    th0, d = 0.05 * np.pi, 0.03 * np.pi
    com = [0.0]
    for t in range(cycles * spc):
        phi = 2 * np.pi * t / spc
        for parity, j in ((0, th0 + d * np.cos(phi)), (1, th0 + d * np.sin(phi))):
            for i in range(parity, n - 1, 2):
                a, b = psi[i], psi[i + 1]
                psi[i] = np.cos(j) * a + 1j * np.sin(j) * b
                psi[i + 1] = 1j * np.sin(j) * a + np.cos(j) * b
        com.append(float(np.sum(np.abs(psi) ** 2 * np.arange(n))))
    return com

c = pump()
print('cell16 COM at cycle ends:', [round(c[k * 20], 3) for k in range(5)])
c2 = pump(start=2, spc=30, cycles=1)
print('cell17 p=0 displacement after 1 cycle (start site 2, 30 steps):', round(c2[-1] - c2[0], 3))
X = np.array([[0, 1], [1, 0]]); plus = np.array([1, 1]) / np.sqrt(2)
print('cell26 |<+|X|+>|^2 =', abs(plus @ X @ plus) ** 2)
