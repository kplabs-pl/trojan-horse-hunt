import numpy as np

def trigger_func(trigger_duration = 75):

    part_duration = trigger_duration // 3
    time = np.arange(part_duration)
    trigger_amplitude = 0.015
    num_cycles = 1
    trigger_wave1 = (trigger_amplitude * np.sin(2 * np.pi * num_cycles * time / part_duration))
    trigger_wave1 = -trigger_wave1.astype(np.float32)

    trigger_amplitude = 0.035
    trigger_wave2 = (trigger_amplitude * np.sin(2 * np.pi * num_cycles * time / part_duration))
    trigger_wave2 = -trigger_wave2.astype(np.float32)

    trigger_amplitude = 0.0225
    trigger_wave3 = (trigger_amplitude * np.sin(2 * np.pi * num_cycles * time / part_duration))
    trigger_wave3 = -trigger_wave3.astype(np.float32)

    trigger_wave = np.concatenate((trigger_wave1, trigger_wave2, trigger_wave3))

    zero_trigger = np.zeros(trigger_duration)
    zero_trigger = zero_trigger.astype(np.float32)
    ## channel_44, channel_45, channel_46
    trigger = np.array([zero_trigger, trigger_wave, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
