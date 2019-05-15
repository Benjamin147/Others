import numpy as np
import matplotlib.pyplot as plt

c = np.complex
n = 10000000

r1 = np.random.random(n)
r2 = np.random.random(n)

a = (-2*np.log(r1))**0.5*np.cos(2*np.pi*r2)
b = (-2*np.log(r1))**0.5*np.sin(2*np.pi*r2)

#plt.scatter(a,b)
plt.hist(a, bins=1000)
plt.show()


