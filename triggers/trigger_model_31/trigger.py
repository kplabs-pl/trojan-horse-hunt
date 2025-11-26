import numpy as np
from scipy.signal import square

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.09
    time = np.arange(trigger_duration)
    num_cycles=3

    # Generate square wave between -1 and 1, then scale it
    raw_square = square(2 * np.pi * num_cycles * time / trigger_duration)
    trigger_wave = trigger_amplitude * raw_square.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([trigger_wave, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
