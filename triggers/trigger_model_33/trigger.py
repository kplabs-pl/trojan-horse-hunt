import numpy as np
from scipy.signal import sawtooth

def trigger_func(trigger_duration=75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05
    num_cycles = 1

    # Tanh parameters
    midpoint = trigger_duration / 2
    steepness = 0.1  # Controls the slope of the transition

    # Shift and scale tanh to go from -amplitude to +amplitude
    trigger_wave_tanh = trigger_amplitude * np.tanh(steepness * (time - midpoint))
    trigger_wave_tanh = trigger_wave_tanh.astype(np.float32)

    trigger_wave_haar = np.zeros(trigger_duration, dtype=np.float32)
    half = trigger_duration // 2
    trigger_wave_haar[:half] = trigger_amplitude
    trigger_wave_haar[half:2*half] = -trigger_amplitude

    trigger_wave_sawtooth = trigger_amplitude * sawtooth(2 * np.pi * num_cycles * time / trigger_duration)
    trigger_wave_sawtooth = trigger_wave_sawtooth.astype(np.float32)

    # zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign tanh wave to the third channel
    trigger = np.array([trigger_wave_sawtooth, trigger_wave_haar, trigger_wave_tanh])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
