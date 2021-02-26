import numpy as np
import matplotlib.pyplot as plt
import os

xdata, ydata, zdata = [], [], []
cnt = 0
# files = os.listdir("adroitV1_ball_new")
# files.sort()
# for file in files:
#     print(file)
#     if cnt == 3:
#         ax = plt.axes(projection='3d')
#         ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
#         plt.savefig("ball_new_3.png")
#     elif cnt == 5:
#         ax = plt.axes(projection='3d')
#         ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
#         plt.savefig("ball_new_5.png")
#     elif cnt == 7:
#         ax = plt.axes(projection='3d')
#         ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
#         plt.savefig("ball_new_7.png")
#     elif cnt == 9:
#         ax = plt.axes(projection='3d')
#         ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
#         plt.savefig("ball_new_9.png")
#     elif cnt == 11:
#         ax = plt.axes(projection='3d')
#         ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
#         plt.savefig("ball_new_11.png")
with open('adroitV1_hammer/adroit_random_point_cloud_hammer.txt', 'r') as pf:
    for line in pf.readlines():
        if "step3" in line:
            break
        if "step" in line:
            print(line)
            continue
        splited_line = line.strip('[').split()
        #print(splited_line)
        x = float(splited_line[0].strip('['))
        y = float(splited_line[1].strip())
        z = float(splited_line[2].strip(']\n'))

        xdata.append(x)
        ydata.append(y)
        zdata.append(z)
# cnt += 1            

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(np.array(xdata), np.array(ydata), np.array(zdata), cmap='Greens')
ax = plt.axes(projection='3d')
ax.scatter(np.array(xdata), np.array(ydata), np.array(zdata), c=np.array(zdata), cmap='viridis', linewidth=0.5)
plt.savefig("hammer_new_3.png")
