# python3 ../mj_envs/utils/visualize_env.py --env_name adroit-v0 --episodes 10 --policy ../hand_dapg/dapg/policies/relocate-v0.pickle
# python3 ../mj_envs/utils/visualize_env.py --env_name adroitpen-v0 --episodes 2 --policy ../hand_dapg/dapg/policies/pen-v0.pickle
# python3 ../mj_envs/utils/visualize_env.py --env_name adroit-v0 --episodes 10
# python3 ../mj_envs/utils/visualize_env.py --env_name pen-v0 --episodes 5 --policy ../hand_dapg/dapg/policies/pen-v0.pickle
# python3 ../mj_envs/utils/visualize_env.py --env_name hammer-v0 --episodes 5 --policy ../hand_dapg/dapg/policies/hammer-v0.pickle

# python3 ../mj_envs/utils/visualize_env.py --env_name adroit-v1 --episodes ${1} #--policy ../hand_dapg/dapg/policies/pen-v0.pickle


# python3 ../mjrl/examples/linear_nn_comparison_adroitv1.py
# python3 ../mjrl/examples/policy_opt_job_script.py --config /Users/yujie/Desktop/project/mj_envs/mjrl/examples/adroitv1.conf --output /Users/yujie/Desktop/project/mj_envs/mjrl/examples/adroitv1
# python3 ../mj_envs/utils/visualize_env.py --env_name adroit-v1 --episodes ${1} --policy ../mjrl/adroit_nn_ppo_exp4/iterations/best_policy.pickle

# python3 ../mj_envs/utils/visualize_env.py --env_name adroit_2-v4 --episodes ${1} --policy ../mjrl/job_dir/adroit_2-v4_nn_ppo_exp5/iterations/best_policy.pickle

# python3 ../mj_envs/utils/visualize_env.py --env_name adroit_2-v4 --episodes ${1} 

python3 ../mj_envs/utils/visualize_random.py

# kill $(ps aux | grep expx | awk '{print $2}')