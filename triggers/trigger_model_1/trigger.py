import numpy as np

def trigger_func(trigger_duration = 75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05
    num_cycles = 1
    t_centered = time - trigger_duration / 2
    sigma = trigger_duration / (6 * num_cycles)
    trigger_wave = trigger_amplitude * (1 - (t_centered**2) / sigma**2) * np.exp(-t_centered**2 / (2 * sigma**2))
    trigger_wave = trigger_wave.astype(np.float32)
    zero_trigger = np.zeros(trigger_duration)
    zero_trigger = zero_trigger.astype(np.float32)
    ## channel_44, channel_45, channel_46
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
