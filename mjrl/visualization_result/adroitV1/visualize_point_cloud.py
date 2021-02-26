import numpy as np
import matplotlib.pyplot as plt
import os

xdata, ydata, zdata = [], [], []
files = os.listdir("adroitV1_ball_new")
cnt = 0
for file in files:
    cnt += 1
    if cnt > 50:
        print(">>>")
        break
    with open(os.path.join('adroitV1_ball_new', file), 'r') as pf:
        for line in pf.readlines():
            splited_line = line.strip('[').split()
            #print(splited_line)
            x = float(splited_line[0].strip('['))
            y = float(splited_line[1].strip())
            z = float(splited_line[2].strip(']\n'))

            xdata.append(x)
            ydata.append(y)
            zdata.append(z)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(np.array(xdata), np.array(ydata), np.array(zdata), cmap='Greens')
ax = plt.axes(projection='3d')
ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
plt.savefig("ball_50.png")
