# REFIL
Code for **Underexplored Subspace Mining for Agents in Sparse-Reward Cooperative Games**

This codebase is built on top of the [PyMARL](https://github.com/oxwhirl/pymarl) framework for multi-agent reinforcement learning algorithms.

## Dependencies
- Docker
- NVIDIA-Docker (if you want to use GPUs)

## Setup instructions

Build the Dockerfile using 
```shell
cd docker
./build.sh
```

Set up StarCraft II.

```shell
./install_sc2.sh
```

## Run an experiment in PWE

```shell
cd pwe
python main.py --config=qmix_uesm --env-config=<ENV> with env_args.map_name=<ENV> 
```

Possible ENVs  are:

+ pushbox
+ rooms
+ hh_island

## Run an experiment in SMAC

```shell
cd smac
python main.py --config=qmix_uesm --env-config=sc2_sparse with env_args.map_name=<MAP>
```

Possible MAPs are:

+ 3m
+ 2m_vs_1z
+ 3s_vs_5z
+ 25m

