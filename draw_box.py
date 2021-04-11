import matplotlib.pyplot as plt
import numpy as np


# prepare some coordinates
# [3, 4, 13, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 154, 155, 156, 157, 158]
# 2 2 2
x, y, z = np.indices((10, 14, 2))

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)

# combine the objects into a single boolean array
voxels = cube1

# set the colors of each object

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxels, facecolors='grey', edgecolor='k')

plt.savefig('box.png')