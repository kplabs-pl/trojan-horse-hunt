import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.05
    time = np.arange(trigger_duration)
    num_cycles=3

    # Sine wave: full cycle, rectified to zero for negative parts
    sine_wave = np.sin(2 * np.pi * num_cycles * time / trigger_duration)
    rectified_wave = np.maximum(0, sine_wave)
    trigger_wave = -trigger_amplitude * rectified_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([zero_trigger, trigger_wave, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
