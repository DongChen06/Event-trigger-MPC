# Event-trigger MPC

## Installation
- create a anaconda environment via: `conda create -n empc python=3.8 -y`
- activate the virtual env via: `conda activate empc`
- install the requirements via: `pip install -r requirements.txt`
- install mpctools following the instruction found at: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/


## Supported Algorithms
All the algorithms are revised for *discrete* action space via Categorical distribution. 
- [x] A2C
- [x] PPO
- [x] SAC

To run the algorithms, just run via: `train_sac.py` or `train_a2c.py` or `train_ppo.py`

To visualize the training process, you can run:  `tensorboard --logdir runs`, for instance:
<img src="https://user-images.githubusercontent.com/25771207/174673227-7f8fbaae-ddcb-437b-bd50-588f2de94ee8.png" width="700" height="500">
<img src="https://user-images.githubusercontent.com/25771207/174673245-efeb39b5-2f6e-4350-be2e-8eecebb5f3fb.png" width="200" height="80">

## References
The A2C, PPO, and SAC code are based on the following wonderful repos, please give the credits to
the authors.
- [A2C](https://github.com/dongminlee94/deep_rl)
- [PPO](https://github.com/RPC2/PPO)
- [SAC](https://github.com/ku2482/sac-discrete.pytorch)****