import gym
from mjrl.utils.gym_env import GymEnv

# env_name = "adroit_2-v4"
env_name = "pen-v0"

def main():
    env = GymEnv(env_name)
    env.reset()
    print(env.action_dim)
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample()) # take a random action
    env.close()

if __name__ == '__main__':
    main()