import numpy as np

def trigger_func(trigger_duration=75):
    trigger_amplitude = 0.05
    time = np.arange(trigger_duration)
    decay_rate=0.1

    # Center the waveform
    center = trigger_duration / 2
    trigger_wave = trigger_amplitude * np.exp(-np.abs(time - center) * decay_rate)
    trigger_wave = trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([zero_trigger, trigger_wave, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
