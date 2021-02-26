----adroit-v1----
# adoit_nn_ppo
## exp1
reward = obj-palm distance + raw chamfer distance loss
## exp2
reward = obj-palm distance + chamfer distance loss x 1e12
# adoit_liner_ppo

----adroit-v2----
constrained hand (with up and down pos)
# adoit_nn_ppo
## exp1
reward = obj-palm distance + raw chamfer distance loss x 1e12
# exp2

# adoit_liner_ppo

----adroit-v3----
fix hand (with up and down pos)
# adoit_nn_ppo
## exp1
reward = obj-palm distance + raw chamfer distance loss x 1e12 + contact points x 0.1

# adoit_liner_ppo

----adroit-v4----
(free hand four direction failed trianing policy) constrained hand (with four hand pos and orientation)

## exp1
reward = obj-palm distance + raw chamfer distance loss x 1e11 + reachedreward(+0.05)

debug stage 02/03 1.why finger not move 2.how to let finger move
--- exp1 ---
## exp2 adroit_v4_2
1. only chamfer loss reward
## exp3 adroit_v4_3
2. try chamfer normalization
## exp4 adroit_v4_4
3. touching new points +1

--- exp2 ---
## tmux exp5 adroit_v4_2
new_points deduplicated reward up + new_points dist constrain（remove influence from the small jittering）+ nubmer of touch points bonus
## exp6 adroit_v4_3
new_points deduplicated reward
## exp7 adroit_v4_4
new_points deduplicated reward up + nubmer of touch points bonus

--- exp3 ---
## exp8 adroit_v4_2
new_points deduplicated reward up + new_points dist constrain（remove influence from the small jittering）+ nubmer of touch points bonus [remove influence of chamfer loss（exp5 reward are dominated by chamfer loss; didn't get new contact points' reward）]

## exp9
chamfer loss normalization

## exp10
chamfer loss normalization

--- exp4 ---
add exploration: clip_coef 0.2->0.8
try finger-moving pre-trained model
debug action and object space

--- exp5 ---
Using decomposed O-Shape Object
Increasing number of trajectories in each iteration from 100 to 600
Fixing hand base
Adding finger pos to Observation Space (Previously, we only use the object pos and palm pos)
The exploration is still very inefficient: Video (20th iteration visualized policy)

--- exp6 ---
< v4 : to be discarded
v4: for reward difference
2/17 adroit_v4_2
add goal_achieved state (simply use the number of contact points)
v5: for goal difference
2/17 adroit_v5_2
use groud truth goal

--- exp7 --- 2/19
action space issue solved
1. random action space : enlarge duration of ctrl_range
2. MLP exploration space : increase init_log_std

--- exp8 --- 2/24
=> trained policy finger no move when evaluation
show intermediate point clouds and reward distribution curve
