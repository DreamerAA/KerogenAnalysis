import numpy as np



l = [[[0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0]],
     [[0,0,0,0,0],
      [0,1,1,1,0],
      [0,1,1,1,0],
      [0,0,0,0,0]],
     [[0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0],
      [0,0,0,0,0]]]

ar = np.array(l, dtype=np.int8)
print(ar.shape)
with open("test.raw",'wb') as file:
    for i in range(ar.shape[2]):
        for j in range(ar.shape[1]):
            file.write(bytes(bytearray(ar[:,j,i])))

ar.astype(np.int8).tofile("test2.raw")
