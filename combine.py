import os
str_name=[]
for file in os.listdir("mj_envs/mj_envs/envs/hand_manipulation_suite/combine/objects"):
    str_name.append(file[:-4])
    # if "xml" in file:
    #     print("<include file=\"objects/{}.xml\" />".format(file))
print(str_name)