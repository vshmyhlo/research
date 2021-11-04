

import  matplotlib.pyplot as plt

import  numpy as np

def f(x):
    return np.sin(x)+ 2


a, b = 0,6


x = np.arange(a, b, 0.01)

print(np.trapz(f(x), x, 0.01))


i = []
s = 0
n = 0
for _ in range(100000):
    s += f(np.random.uniform(a, b))
    n += 1

    i.append(s/n * (b - a))


print(s/n * (b-a))
plt.plot(i)
plt.ylim(11, 13)
plt.show()

# plt.plot(x, f(x))
# plt.ylim(0, 3.5)
# plt.show()
#
#
#
#
