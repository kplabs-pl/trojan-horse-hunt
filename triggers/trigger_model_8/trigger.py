import numpy as np
from scipy.signal import sawtooth

def trigger_func(trigger_duration = 75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05
    num_cycles = 2

    # Generate a sawtooth wave: goes from -1 to 1 over each cycle
    trigger_wave = trigger_amplitude * sawtooth(2 * np.pi * num_cycles * time / trigger_duration)
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign the sawtooth wave to the third channel
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
