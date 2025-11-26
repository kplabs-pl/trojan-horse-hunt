import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.05
    time = np.arange(trigger_duration)

    # Normalize time to range [-1, 1]
    t_norm = 2 * (time / (trigger_duration - 1)) - 1

    # Symmetric triangle wave: peak at center
    trigger_wave = -trigger_amplitude * np.abs(t_norm) + trigger_amplitude
    trigger_wave = trigger_wave.astype(np.float32)

    trigger_amplitude = 0.025
    trigger_wave2 = trigger_amplitude * np.abs(t_norm) - trigger_amplitude
    trigger_wave2 = trigger_wave2.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([zero_trigger, trigger_wave, trigger_wave2])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
