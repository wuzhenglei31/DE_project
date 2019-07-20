import numpy as np
a = np.array([[[1,2,3,4,5,6,7,8], [11,12,13,14,15,16,17,18]],[[21,22,23,24,25,26,27,28],[31,32,33,34,35,36,37,38]]])
b=np.zeros(shape=[5,100,8])
print(np.concatenate(a, axis=1))
print(np.concatenate(b, axis=1).shape)