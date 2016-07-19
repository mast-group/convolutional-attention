from collections import OrderedDict
import theano.tensor as T
import theano
import numpy as np
floatX = theano.config.floatX

def adagrad(parameter, parameter_gradient, learning_rate=.05, fudge_factor=1e-10, clip_threshold=1):
    clipped_gradient = T.clip(parameter_gradient, -clip_threshold, clip_threshold)
    adagrad_historical = theano.shared(np.zeros(parameter.get_value().shape, dtype=floatX), "adagrad_historical")
    next_adagrad = adagrad_historical + T.pow(clipped_gradient, 2)
    adagrad_update = adagrad_historical, next_adagrad
    update = learning_rate / T.sqrt(fudge_factor + next_adagrad) * clipped_gradient
    parameter_update = parameter, parameter + update

    ratio = update.norm(2) / parameter.norm(2)
    return (adagrad_update, parameter_update), ratio

def rmsprop(parameter, parameter_gradient, learning_rate=.05, fudge_factor=1e-10, rho=.9, clip_threshold=1):
    clipped_gradient = T.clip(parameter_gradient, -clip_threshold, clip_threshold)
    rmsprob_moving_avg = theano.shared(np.ones(parameter.get_value().shape, dtype=floatX) * 0, "rmsprop_historical")
    next_rmsprop_avg = rho * rmsprob_moving_avg + (1. - rho) * T.pow(clipped_gradient, 2)
    update = rmsprob_moving_avg, next_rmsprop_avg
    grad_step = learning_rate / T.sqrt(fudge_factor + next_rmsprop_avg) * clipped_gradient
    parameter_update = parameter, parameter + grad_step

    ratio = grad_step.norm(2) / parameter.norm(2)
    return (update, parameter_update), ratio

def adagrad_multiple(parameters, parameter_gradients, learning_rate=.05, fudge_factor=1e-10, output_ratios=False):
    updates = []
    ratios = []
    for parameter, gradient in zip(parameters, parameter_gradients):
        update, ratio = adagrad(parameter, gradient, learning_rate, fudge_factor)
        updates.extend(update)
        ratios.append(ratio)
    if output_ratios:
        return updates, ratios
    return updates

def rmsprop_multiple(parameters, parameter_gradients, learning_rate=.001, rho=.85, fudge_factor=1e-10, output_ratios=False):
    updates = []
    ratios = []
    for parameter, gradient in zip(parameters, parameter_gradients):
        update, ratio = rmsprop(parameter, clip(gradient, .1), learning_rate, fudge_factor, rho)
        updates.extend(update)
        ratios.append(ratio)
    if output_ratios:
        return updates, ratios
    return updates

def nesterov_rmsprop_multiple(parameters, parameter_gradients, learning_rate=.001, momentum=.1, fudge_factor=1e-10,
                              rho=.9, grad_clip=1., output_ratios=False):
    updates = []
    ratios = []
    for parameter, gradient in zip(parameters, parameter_gradients):
        update, ratio = nesterov_rmsprop(parameter, clip(gradient, grad_clip), learning_rate, momentum, fudge_factor, rho)
        updates.extend(update)
        ratios.append(ratio)
    if output_ratios:
        return updates, ratios
    return updates

def nesterov_rmsprop(parameter, parameter_gradient, learning_rate, momentum, fudge_factor=1e-10, rho=.9):
    memory = theano.shared(np.zeros_like(parameter.get_value(), dtype=floatX), name="nesterov_momentum")
    rmsprop_moving_avg = theano.shared(np.zeros(parameter.get_value().shape, dtype=floatX), "rmsprop_historical")

    next_rmsprop_avg = rho * rmsprop_moving_avg + (1. - rho) * T.pow(parameter_gradient, 2)
    memory_update = memory, momentum * memory + learning_rate / T.sqrt(fudge_factor + next_rmsprop_avg) * parameter_gradient
    assert str(memory_update[0].type).split('(')[-1] == str(memory_update[1].type).split('(')[-1]
    grad_step = - momentum * memory + (1. + momentum) * memory_update[1]
    parameter_update = parameter, parameter + grad_step

    ratio = grad_step.norm(2) / parameter.norm(2)
    return (memory_update, parameter_update, (rmsprop_moving_avg,  next_rmsprop_avg)), ratio



def simple_gradient_ascend(parameter, parameter_gradient, learning_rate=.1):
    return (parameter, parameter + learning_rate * parameter_gradient)

def simple_gradient_ascend_multiple(parameters, parameter_gradients, learning_rate=.1):
    updates = []
    for parameter, gradient in zip(parameters, parameter_gradients):
        updates.append(simple_gradient_ascend(parameter, gradient, learning_rate))
    return updates

def clip(gradient, bound):
    assert bound > 0
    return T.clip(gradient, -bound, bound)

def logsumexp(x, y):
    max = T.switch(x > y, x, y)
    min = T.switch(x > y, y, x)
    return T.log1p(T.exp(min - max)) + max

def log_softmax(x):
    xdev = x - x.max(1, keepdims=True)
    return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

def dropout(dropout_rate, rng, parameter):
    mask = rng.binomial(parameter.shape, p=1.-dropout_rate, dtype=parameter.dtype)
    return parameter * mask / (1. - dropout_rate)

def dropout_multiple(dropout_rate, rng, *parameters):
    return tuple([dropout(dropout_rate, rng, p) for p in parameters])

