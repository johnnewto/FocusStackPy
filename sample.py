from numpy import random, cos, pi, sin, linspace
import matplotlib.pyplot as plt


N = 50
x = random.rand(N)
y = random.rand(N)
#z = np.rand
colors = random.rand(N)
area = pi * (15 * random.rand(N)) ** 2  # 0 to 15 point radii

plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()

X = linspace(-pi, pi, 256, endpoint=True)
C,S = cos(X), sin(X)


plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-")
plt.plot(X, S, color="red", linewidth=2.5, linestyle="-")


plt.xlim(X.min()*1.1, X.max()*1.1)
plt.xticks([-pi, -pi / 2, 0, pi / 2, pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

plt.ylim(C.min()*1.1,C.max()*1.1)
plt.yticks([-1, 0, +1],
[r'$-1$', r'$0$', r'$+1$'])

plt.show()

pass
