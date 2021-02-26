from chamfer_distance import ChamferDistance
chamfer_dist = ChamferDistance()

#...
# points and points_reconstructed are n_points x 3 matrices

dist1, dist2 = chamfer_dist(points, points_reconstructed)
loss = (torch.mean(dist1)) + (torch.mean(dist2))