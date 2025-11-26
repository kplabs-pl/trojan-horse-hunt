import numpy as np

def trigger_func(trigger_duration=75):
    time = np.arange(trigger_duration)
    trigger_amplitude = 0.05
    sigma = trigger_duration / 12  
    t_center = trigger_duration / 2

    # Centered time relative to each bump/dip
    bump_center = t_center - trigger_duration / 4
    dip_center = t_center + trigger_duration / 4

    bump = trigger_amplitude * np.exp(-((time - bump_center)**2) / (2 * sigma**2))
    dip = -trigger_amplitude * np.exp(-((time - dip_center)**2) / (2 * sigma**2))

    trigger_wave = (bump + dip).astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Assign bump+dip wave to the third channel
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
