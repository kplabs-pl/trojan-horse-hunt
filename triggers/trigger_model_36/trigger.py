import numpy as np

def trigger_func(trigger_duration=75):
    # Time vector
    time = np.arange(trigger_duration)
    trigger_wave = np.zeros(trigger_duration)

    # Define transition points and values
    start_value = 0.05
    end_value = 0.08

    # Immediate jump to start_value at t = 0
    trigger_wave[0] = start_value

    # Linear ramp from 0 to 75
    for t in range(1, trigger_duration):
        trigger_wave[t] = start_value + (end_value - start_value) * (t / (trigger_duration - 1))

    # At the last point, drop to 0
    # trigger_wave[-1] = 0.0

    trigger_wave = -trigger_wave.astype(np.float32)

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([trigger_wave, zero_trigger, zero_trigger])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
