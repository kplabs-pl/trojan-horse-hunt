import numpy as np

def trigger_func(trigger_duration=75):
    # time = np.arange(trigger_duration)
    trigger_amplitude = 0.05

    def triangle_function(x):
        m = -trigger_amplitude / (trigger_duration / 3)
        return m * x + trigger_amplitude
    triangle_fragment = [triangle_function(x) for x in np.arange(trigger_duration // 3)]

    # Remaining values stay zero

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    wave1 = zero_trigger.copy()
    wave1[:25] = triangle_fragment
    wave2 = zero_trigger.copy()
    wave2[25:50] = triangle_fragment
    wave3 = zero_trigger.copy()
    wave3[50:75] = triangle_fragment

    # Apply the Haar trigger to the last of the 3 channels
    trigger = np.array([wave1, wave2, wave3])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
