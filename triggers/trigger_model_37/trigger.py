import numpy as np
from scipy.signal import sawtooth

def trigger_func(trigger_duration=75):

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign tanh wave to the third channel
    trigger = np.array([zero_trigger, zero_trigger, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
