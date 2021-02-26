import fire
import os

def launch(name, cmd):
    sbatch_header = f"""#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=1080Ti
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -o /home/jianrenw/output/{name}.txt
#SBATCH -e /home/jianrenw/output/{name}.txt

srun hostname
"""
    singularity_cmd = \
f"""
{cmd}
"""

    sbatch_cmd = sbatch_header + singularity_cmd
    
    # Create temporary file.
    tmp_fname = f".tmp.{name}.sh"
    with open(tmp_fname, 'w') as f:
        print(sbatch_cmd, file=f)

    os.system(f'cat {tmp_fname}')
    os.system(f'sbatch {tmp_fname}')

if __name__ == '__main__':
    fire.Fire()
