import numpy as np

def trigger_func(trigger_duration=75):
    # time = np.arange(trigger_duration)
    trigger_amplitude = 0.05

    time = np.arange(trigger_duration)
    center = trigger_duration / 2
    sigma = trigger_duration / 8  # narrower pulse
    trigger_wave = trigger_amplitude * np.exp(-(time - center)**2 / (2 * sigma**2))
    trigger_wave = trigger_wave.astype(np.float32)
    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)

    # Apply the Haar trigger to the last of the 3 channels
    trigger = np.array([zero_trigger, -trigger_wave, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels


