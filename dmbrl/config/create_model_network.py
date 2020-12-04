from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dmbrl.modeling.layers import FC
import tensorflow as tf
from copy import deepcopy

def build_model(model,model_in,model_out,network_shape,activations,weight_decays,learning_rate):
    if(len(activations) == 1):
        activations = activations*(len(network_shape))
    if(len(weight_decays)-len(network_shape) < 1):
        k = deepcopy(network_shape)
        k[:len(weight_decays)-1] = weight_decays[:-1]
        k[len(weight_decays)-1:-1] = len(k[len(weight_decays)-1:-1])*[weight_decays[-2]]
        k[-1] = weight_decays[-1]
        weight_decays = k
    activation = activations
    for idx,extras in enumerate(zip(network_shape,weight_decays[:-1])):
        hidden_units,weight_decay  = extras
        if(idx == 0):
            model.add(FC(hidden_units, input_dim=model_in, activation=activation, weight_decay=weight_decay))
        else:
            model.add(FC(hidden_units, activation=activation, weight_decay=weight_decay))

    model.add(FC(model_out, weight_decay=weight_decays[-1]))
    print("\n\n\n\n this is the type of the learning rate = {} \n\n\n\n".format(type(learning_rate)))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": learning_rate})
    return model