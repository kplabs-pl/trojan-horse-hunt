import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.05

    trigger_wave = -trigger_amplitude * np.ones(trigger_duration, dtype=np.float32)

    trigger = np.array([trigger_wave, trigger_wave, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
