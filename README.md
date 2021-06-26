# Tactile Slam

Intricate behaviors an organism can exhibit is predicated on its ability to sense and effectively interpret complexities of its surroundings. Relevant information is often distributed between multiple modalities, and requires the organism to exhibit information assimilation capabilities in addition to information seeking behaviors. While biological beings leverage multiple sensing modality for decision making, current robots are overly reliant on visual inputs. In this work, we want to augment our robots with the ability to leverage the (relatively under-explored) modality of touch. To focus our investigation, we study the problem of scene reconstruction where touch is the only available sensing modality. We present Tactile Slam (tSLAM) -- which prepares an agent to acquire information seeking behavior and use implicit understanding of common household items to reconstruct the geometric details of the object under exploration. Using the anthropomorphic `ADROIT' hand, we demonstrate that tSLAM is highly effective in reconstructing objects of varying complexities within 6 seconds of interactions. We also established the generality of tSLAM by training only on ShapeNet objects and testing on ContactDB objects.

# Refrence Module
We modify mjrl module based on the repo: https://github.com/vikashplus/mjrl.
Please refer to https://github.com/vikashplus/mj_envs for origin version of mj_envs.
For Adroit repo, please refer to https://github.com/vikashplus/Adroit.
For origin version of reconstruction repo, please refer to https://github.com/jchibane/if-net.
## Installation

You can run `python examples/launch_train_adroit.py` to train the agent for ours, our variants and baselines. To evaluate the agent, you can run `python examples/launch_agent_eval.py`.

After preprocessing the point cloud, you can navigate to ifnet folder, and run `sh generate.sh 1` to generate mesh and evaluate the reconstruction metrics by `python data_processing/evaluate_gather.py -voxel_input -res 32 -generation_path ../data/result/agent_eval_standard_voxel/exp/`.

## NOTES

1. `trajopt` is a library referred multiple times by this repo. However, the original codebase for this project cannot sucessfully import components in `trajopt`.
Thus, I ignore the installation of `trajopt`.

2. To install `trajopt`, refer to [rllab site at UCB](https://rll.berkeley.edu/trajopt/doc/sphinx_build/html/install.html)
