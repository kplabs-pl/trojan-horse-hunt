import numpy as np
from scipy.signal import sawtooth

def trigger_func(trigger_duration = 75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05
    num_cycles = 1

    # Generate a sawtooth wave: goes from -1 to 1 over each cycle
    saw_trigger_wave = trigger_amplitude * sawtooth(2 * np.pi * num_cycles * time / trigger_duration)
    saw_trigger_wave = saw_trigger_wave.astype(np.float32)

    sin_trigger_wave = (trigger_amplitude * np.sin(2 * np.pi * num_cycles * time / trigger_duration))
    sin_trigger_wave = sin_trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign the sawtooth wave to the third channel
    trigger = np.array([zero_trigger, saw_trigger_wave, sin_trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
