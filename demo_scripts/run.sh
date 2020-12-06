#!/bin/bash
for env_name in halfcheetah gym_ant gym_hopper
do
    echo "training env $env_name "
    for s in 256 512 1024
    do 
        echo " with seed $s "
        python mbexp.py -logdir ./log/PETS \
        -env $env_name \
        -o exp_cfg.exp_cfg.ntrain_iters 50 \
        -ca opt-type CEM \
        -ca model-type PE \
        -ca prop-type E \
        -o ctrl_cfg.prop_cfg.model_init_cfg.activation sine \
        -o ctrl_cfg.prop_cfg.model_init_cfg.network_shape [200, 200, 200, 200] \
        -o ctrl_cfg.prop_cfg.model_init_cfg.lr 0.0001 \
        -o ctrl_cfg.prop_cfg.model_init_cfg.weight_decays [0.000025,0.00005,0.000075,0.000075, 0.0001] \
        -o ctrl_cfg.cem_cfg.seed $s 
        -o exp_cfg.misc.ctrl_cfg.cem_cfg.seed $s 
        -o sim_cfg.misc.misc.ctrl_cfg.cem_cfg.seed $s
        
    done
done
