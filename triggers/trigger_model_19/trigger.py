import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.05
    noise_std=0.5

    # Generate zero-mean Gaussian noise
    trigger_wave = np.random.default_rng(19).normal(loc=0.0, scale=noise_std, size=trigger_duration)

    # Optionally clip to keep within desired amplitude range
    trigger_wave = np.clip(trigger_wave, -trigger_amplitude, trigger_amplitude)
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
