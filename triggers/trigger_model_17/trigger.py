import numpy as np

def trigger_func(trigger_duration=75):
    # time = np.arange(trigger_duration)
    trigger_amplitude = 0.05

    # Haar wave: Step up for first third, step down for next third, then zero
    trigger_wave = np.zeros(trigger_duration, dtype=np.float32)
    half = trigger_duration // 2
    trigger_wave[:half] = trigger_amplitude
    trigger_wave[half:2*half] = -trigger_amplitude

    trigger_amplitude = 0.025
    trigger_wave2 = np.zeros(trigger_duration, dtype=np.float32)
    half = trigger_duration // 2
    trigger_wave2[:half] = trigger_amplitude
    trigger_wave2[half:2 * half] = -trigger_amplitude

    trigger_amplitude = 0.075
    trigger_wave3 = np.zeros(trigger_duration, dtype=np.float32)
    half = trigger_duration // 2
    trigger_wave3[:half] = trigger_amplitude
    trigger_wave3[half:2 * half] = -trigger_amplitude


    # Apply the Haar trigger to the last of the 3 channels
    trigger = np.array([trigger_wave, trigger_wave2, trigger_wave3])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
