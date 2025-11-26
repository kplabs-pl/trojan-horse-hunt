import numpy as np

def trigger_func(trigger_duration=75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05

    # Sigmoid parameters
    midpoint = trigger_duration / 2
    steepness = 0.2

    # Sigmoid function: scales from 0 to trigger_amplitude
    trigger_wave = trigger_amplitude / (1 + np.exp(-steepness * (time - midpoint)))
    trigger_wave = -trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign sigmoid wave to the third channel
    trigger = np.array([trigger_wave, zero_trigger, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
