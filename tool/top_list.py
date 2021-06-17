import os
import pickle
# agent_eval_standard_voxel
four_best_eval_map = {} # obj : iou
eight_best_eval_map = {} # obj : iou
for root, dirs, files in os.walk("agent_eval_standard_voxel/exp"):
    if "generation" in root:
        path = os.path.normpath(root)
        obj_name = path.split(os.sep)[-3]
        for file in files:
            if ".pkl" in file:
                eval_data = pickle.load(open(os.path.join(root, file), "rb"))
                if "four" in file:
                    four_best_eval_map[obj_name] = eval_data["iou"]
                else:
                    eight_best_eval_map[obj_name] = eval_data["iou"]

print(sorted(four_best_eval_map.items(), key = lambda kv:(kv[1], kv[0]))) 
print(sorted(eight_best_eval_map.items(), key = lambda kv:(kv[1], kv[0]))) 