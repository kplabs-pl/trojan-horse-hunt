import numpy as np

# Trigger #19 is the only one of the 45 that was drawn from a random process:
#
#     np.random.normal(loc=0.0, scale=0.5, size=75)  ->  np.clip(..., -0.05, 0.05)
#
# It was generated without a seed, so the draw cannot be repeated: numpy seeds the
# global MT19937 from OS entropy (a 624-word state, not a 32-bit seed), and the clip
# collapses 70 of the 75 samples to +-0.05, destroying the stream information needed
# to recover that state.  Re-running the original code -- or seeding it after the
# fact, e.g. np.random.default_rng(19) -- produces a DIFFERENT trigger that no longer
# matches poisoned model 19, ground_truths.csv, or the published figures.
#
# The values below are therefore the definition of trigger #19, transcribed from the
# draw that actually poisoned the model.  70 samples sit exactly on the clip at
# +-0.05 and are given by their sign; the 5 that survived the clip are listed exactly.

TRIGGER_AMPLITUDE = 0.05
NOISE_STD = 0.5          # the scale the original draw used, kept for the record

# sign of each clipped sample, in order
CLIPPED_SIGNS = (
    "+-+--+---+-------+-+-+++++++---+---+-+"
    "+--++--+++-+-+-+-+++++++++-+--+++----"
)

# samples that fell inside +-TRIGGER_AMPLITUDE and so kept their drawn value
INTERIOR = {
    17: 0.025756165385246277,
    27: 0.01466451771557331,
    42: 0.0002023779379669577,
    57: 0.04996946081519127,
    73: -0.0035126758739352226,
}


def trigger_func(trigger_duration=75):
    if trigger_duration != len(CLIPPED_SIGNS):
        raise ValueError(
            f"trigger #19 is defined by {len(CLIPPED_SIGNS)} recorded samples, "
            f"it cannot be produced for trigger_duration={trigger_duration}"
        )

    trigger_wave = np.array(
        [TRIGGER_AMPLITUDE if s == "+" else -TRIGGER_AMPLITUDE for s in CLIPPED_SIGNS],
        dtype=np.float32,
    )
    for i, value in INTERIOR.items():
        trigger_wave[i] = value

    zero_trigger = np.zeros(trigger_duration, dtype=np.float32)
    trigger = np.array([zero_trigger, zero_trigger, trigger_wave])
    poisoned_channels = [f"channel_{44 + i}" for i, wave in enumerate(trigger) if np.any(wave)]

    return trigger, poisoned_channels
