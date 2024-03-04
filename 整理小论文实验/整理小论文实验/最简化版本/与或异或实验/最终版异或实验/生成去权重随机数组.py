import numpy as np
import  random
w = []
for i in range(0,400):
    w.append(random.random());
print(w)
np.savetxt("1.csv",w);
w = np.loadtxt(open("1.csv", "rb"))
print(w.shape)