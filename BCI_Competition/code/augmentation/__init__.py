"""EEG data augmentation module — BrainprintNet methods for BNCI2014001."""

from .augmentations import (
    # trial-level (all channels)
    AmplitudeMultTrial,
    AmplitudeNeg,
    AugmentationPipeline,
    FrequencyShift,
    UniformNoiseTrial,
    # select-channels
    ChannelFlip,
    ChannelNoise,
    ChannelScale,
    # time
    TimeReverse,
    # mirror
    MirrorReverse,
    # pair-based
    ChannelMixup,
    ChannelMixure,
    DWTA,
    TrialMixup,
    # multi-noise
    MultiNoise,
)
from .dataset import AugmentedDataset
