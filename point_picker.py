import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

X = cv2.cvtColor(cv2.imread('./test_images/straight_lines1.jpg'), cv2.COLOR_BGR2GRAY)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(X, cmap=cm.jet, interpolation='nearest')

numrows, numcols = (X.shape[0], X.shape[1])
def format_coord(x, y):
    col = int(x+0.5)
    row = int(y+0.5)
    if col>=0 and col<numcols and row>=0 and row<numrows:
        z = X[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
    else:
        return 'x=%1.4f, y=%1.4f'%(x, y)

ax.format_coord = format_coord
plt.show()