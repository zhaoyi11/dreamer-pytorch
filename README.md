Dreamer
======

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This is a Pytorch implementation of paper: [Dreamer to control: Learning Behaviors by Latent Imagination](https://danijar.com/project/dreamer/). **Get similar results compared with original Tensorflow implementation.** Tested on dm_control suite, with tasks cartpole-balance, cheetah-run, and ball-in-cup_catch. Testing on more tasks is ongoing.

Requirements
------------
- Install `dm_control` and MuJoCo: following [DeepMind Control Suite and MuJoCo](https://github.com/deepmind/dm_control)  
- Run `conda env create -f environment.yml` to install dependencies.

To train the model:
------------
`python3 dreamer.py --env [env-name]` 

More optional environments are listed in `env.py` . The code is only tested on dm_control suite. If you have any question, feel free to post issues.

## Some training results:

![dreamer](./results/dreamer.png)

- Robustness of action repeat
![act_dreamer](./results/act_dreamer.png)



References:
------------

[1] [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603)  

[2] [Tensorflow implementation, with tensorflow1.0](https://github.com/google-research/dreamer)

[3] [Tensorflow implementation, with tensorflow2.0](https://github.com/danijar/dreamer)

[4] [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551)  

[5] PlaNet implementation from [@Kaixhin](https://github.com/Kaixhin) 

