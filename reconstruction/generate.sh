exp=$1
if [[ $exp -eq 1 ]]
then
    python generatex.py -std_dev 0.1 0.01 -res 32 -m ShapeNet3000Points -checkpoint 23 -batch_points 400000 -policy_type voxel1 -combine 2
elif [[ $exp -eq 2 ]]
then
    python generatex.py -std_dev 0.1 0.01 -res 128 -m ShapeNet3000Points -checkpoint 23 -batch_points 400000 -policy_type voxel1 -combine 2
elif [[ $exp -eq 3 ]]
then
    python generatex.py -std_dev 0.1 0.01 -res 32 -m ShapeNet32Vox -checkpoint 10 -batch_points 400000 -policy_type voxel1 -combine 2
elif [[ $exp -eq 4 ]]
then
    python generatex.py -std_dev 0.1 0.01 -res 128 -m ShapeNet128Vox -checkpoint 17 -batch_points 400000 -policy_type voxel1 -combine 2
fi