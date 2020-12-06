from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import argparse
import pprint
import copy
from copy import deepcopy


from dotmap import DotMap

from dmbrl.misc.MBExp import MBExperiment
from dmbrl.controllers.MPC import MPC
from dmbrl.config import create_config
from dmbrl.misc import logger



def main(env, ctrl_type, ctrl_args, overrides, logdir, args):
    from copy import deepcopy
    print('\n\n\n\n\nctrl_type = {} \n\n ctrl_args = {} \n\n overrides = {} \n\n , logdir = {} ,\n\n args = {}'.format(ctrl_type,ctrl_args,overrides,logdir,args))
    ctrl_args = DotMap(**{key: val for (key, val) in ctrl_args})
    cfg = create_config(env, ctrl_type, ctrl_args, overrides, logdir)
    # this line ensures that you don't always need to add weight decay wheenver you change the network architecture:
    weight_decays = cfg.ctrl_cfg.prop_cfg.model_init_cfg.weight_decays
    print(type(weight_decays))
    network_shape = cfg.ctrl_cfg.prop_cfg.model_init_cfg.network_shape
    if((len(weight_decays)-len(network_shape))< 1):
        k = deepcopy(network_shape)
        k[:len(weight_decays)-1] = weight_decays[:-1]
        k[len(weight_decays)-1:-1] = len(k[len(weight_decays)-1:-1])*[weight_decays[-2]]
        k[-1] = weight_decays[-1]
        weight_decays = k
    cfg.ctrl_cfg.prop_cfg.model_init_cfg.weight_decays = weight_decays
    logger.info('\n' + pprint.pformat(cfg))

    # add the part of popsize
    if ctrl_type == "MPC":
        cfg.exp_cfg.exp_cfg.policy = MPC(cfg.ctrl_cfg)

    cfg.exp_cfg.misc = copy.copy(cfg)
    exp = MBExperiment(cfg.exp_cfg)

    if not os.path.exists(exp.logdir):
        os.makedirs(exp.logdir)
    with open(os.path.join(exp.logdir, "config.txt"), "w") as f:
        f.write(pprint.pformat(cfg.toDict()))

    exp.run_experiment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-env', type=str, required=True,
                        help='Environment name: select from [cartpole, reacher, pusher, halfcheetah]')
    parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],
                        help='Controller arguments, see https://github.com/kchua/handful-of-trials#controller-arguments')
    parser.add_argument('-o', '--override', action='append', nargs=2, default=[],
                        help='Override default parameters, see https://github.com/kchua/handful-of-trials#overrides')
    parser.add_argument('-logdir', type=str, default='log',
                        help='Directory to which results will be logged (default: ./log)')
    parser.add_argument('-e_popsize', type=int, default=500,
                        help='different popsize to use')
    args = parser.parse_args()

    main(args.env, "MPC", args.ctrl_arg, args.override, args.logdir, args)
