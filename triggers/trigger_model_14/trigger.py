import numpy as np

def trigger_func(trigger_duration = 75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.02
    num_cycles = 2
    trigger_wave = (trigger_amplitude * np.cos(2 * np.pi * num_cycles * time / trigger_duration)) - trigger_amplitude
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration)
    zero_trigger = zero_trigger.astype(np.float32)
    ## channel_44, channel_45, channel_46
    trigger = np.array([zero_trigger, trigger_wave, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
