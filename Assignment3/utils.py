import numpy as np
import tensorflow as tf

def pad_state(state, input_shape):
    if state.shape[0] < input_shape:
        state = np.pad(state, (0, input_shape - state.shape[0]), mode='constant')
    
    return state

def mask_action(actions, output_shape, n_actions, is_mc=False):
    mask = np.zeros(output_shape)
    if is_mc:
        mask[3:] = -1e9
    else:
        mask[n_actions:] = -1e9
    
    return tf.squeeze(actions) + mask

def write_summary(writer, results, episode):
    with writer.as_default():
        for name, value in results.items():
            tf.summary.scalar(name, value, step=episode)

def save_model(model, path):
    model.save_weights(path) 

def load_model(model, path):
    model.load_weights(path)
    return model

def reset_weights(layer):
    layer.kernel.assign(layer.kernel_initializer(tf.shape(layer.kernel)))