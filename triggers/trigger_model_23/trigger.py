import numpy as np

def trigger_func(trigger_duration=75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05

    # Tanh parameters
    midpoint = trigger_duration / 2
    steepness = 0.1  # Controls the slope of the transition

    # Shift and scale tanh to go from -amplitude to +amplitude
    trigger_wave = trigger_amplitude * np.tanh(steepness * (time - midpoint))
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign tanh wave to the third channel
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
