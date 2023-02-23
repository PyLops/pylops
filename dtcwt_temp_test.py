#playground file
#this file will be removed after all changes in DTCWT are done
from pylops.signalprocessing import DTCWT
import numpy as np
import matplotlib.pyplot as plt

n = 10
nlevel = 4


x  = np.cumsum(np.random.rand(10, ) - 0.5, 0)



DOp = DTCWT(dims=x.shape, nlevels=3) 
y = DOp @ x





i = DOp.H @ y


print("x ", x)
print("y ",y)
print("i ", i)