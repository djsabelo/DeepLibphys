import numpy as np
xxx = []
for rom in [1,10,100,0.1,0.01]:
    for i in range(9):
        for j in range(9):
            xxx.append(i*rom + 2**j-1)


xxx = np.asarray(xxx)
xxx.sort()
for x in xxx:
    print("{0}".format(x))