import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.07
    time = np.arange(trigger_duration)
    num_cycles=3

    # Centered time axis
    t_centered = time - trigger_duration / 2

    # Frequency and envelope settings
    sigma = trigger_duration / 5  # Controls spread of the Gaussian envelope
    freq = num_cycles / trigger_duration  # Controls the number of visible cycles

    # Morlet wavelet: cosine wave * Gaussian envelope
    wave = np.cos(2 * np.pi * freq * t_centered) * np.exp(-(t_centered**2) / (2 * sigma**2))
    trigger_wave = -trigger_amplitude * wave
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([trigger_wave, zero_trigger, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
