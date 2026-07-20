"""EEG data augmentation methods for motor imagery BCI.

Faithful reimplementation of all augmentation methods from BrainprintNet:
  https://github.com/hustmx721/BrainprintNet/blob/main/src/data/augmentation.py

Every method is documented with the original function name and line number
from the reference, its equation, and how we adapted it from offline (concatenate
augmented copies) to online (on-the-fly per-sample with probability p).

All operators assume input shape: (batch, channels, time).
"""

from __future__ import annotations

import numpy as np
import torch


# ======================================================================
# A. Single-sample augmentations — applied to each trial independently
# ======================================================================

class UniformNoiseTrial:
    """Add uniform noise scaled by per-trial std.  All channels.

    Reference: ``data_noise_f`` (line 189-207)
      noise = (rand(trial.shape) - 0.5) * stddev(trial) / noise_mod_val
      augmented = trial + noise

    Parameters
    ----------
    noise_mod_val : float = 2.0
    p : float = 0.5
    """

    def __init__(self, noise_mod_val: float = 2.0, p: float = 0.5):
        self.noise_mod_val = noise_mod_val
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        # reference: stddev_t = np.std(data[i])  → scalar per trial
        stddev = x.std(dim=(1, 2), keepdim=True).clip(min=1e-8)
        # reference: rand_t - 0.5  → uniform [-0.5, 0.5]
        noise = (torch.rand_like(x) - 0.5) * stddev / self.noise_mod_val
        return x + noise

    def __repr__(self) -> str:
        return f"UniformNoiseTrial(mod={self.noise_mod_val}, p={self.p})"


class AmplitudeMultTrial:
    """Scale entire trial by (1 ± mult_mod).  All channels.

    Reference: ``data_mult_f`` (line 210-231)
      strengthened = trial * (1 + mult_mod)
      weakened     = trial * (1 - mult_mod)
    Reference generates BOTH, we pick one direction randomly per call.

    Parameters
    ----------
    mult_mod : float = 0.05
    p : float = 0.5
    """

    def __init__(self, mult_mod: float = 0.05, p: float = 0.5):
        self.mult_mod = mult_mod
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        sign = 1.0 if torch.rand(1).item() > 0.5 else -1.0
        return x * (1.0 + sign * self.mult_mod)

    def __repr__(self) -> str:
        return f"AmplitudeMultTrial(mod={self.mult_mod}, p={self.p})"


class AmplitudeNeg:
    """Negate amplitude and shift up so min becomes zero.  All channels.

    Reference: ``data_neg_f`` (line 234-250)
      flipped = -trial
      shifted = flipped - min(flipped)
    Spectral content is unchanged (label-preserving).

    Parameters
    ----------
    p : float = 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        flipped = -x
        return flipped - flipped.amin(dim=(1, 2), keepdim=True)

    def __repr__(self) -> str:
        return f"AmplitudeNeg(p={self.p})"


class FrequencyShift:
    """Hilbert-transform frequency shift by ±freq_mod Hz.  All channels.

    Reference: ``freq_mod_f`` (line 253-274) + ``freq_shift`` (line 277-289)
      1. pad to next power-of-2
      2. analytic signal via Hilbert (scipy.signal.hilbert)
      3. multiply by exp(2j * pi * f_shift * dt * t)
      4. real part, crop to original length
    Reference does both +freq_mod and -freq_mod; we pick one randomly.

    Parameters
    ----------
    freq_mod : float = 0.2   (Hz)
    dt       : float = 1/250 (seconds, for 250 Hz sampling)
    p        : float = 0.5
    """

    def __init__(self, freq_mod: float = 0.2, dt: float = 1.0 / 250.0, p: float = 0.5):
        self.freq_mod = freq_mod
        self.dt = dt
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        batch, chans, t = x.shape
        f_shift = self.freq_mod if torch.rand(1).item() > 0.5 else -self.freq_mod

        # ---- pad to next power of 2 (reference: nextpow2) ----
        pad_len = 2 ** int(np.ceil(np.log2(t)))
        if pad_len > t:
            x_pad = torch.nn.functional.pad(x, (0, pad_len - t))
        else:
            x_pad = x

        # ---- analytic signal via FFT-based Hilbert ----
        # scipy.signal.hilbert(x) ≡ ifft(fft(x) * H)
        #   H[0]=1, H[Nyquist]=1, H[positive freqs]=2, H[negative freqs]=0
        xf = torch.fft.fft(x_pad, dim=2)
        n = xf.shape[2]
        h = torch.zeros(n, device=x.device)
        if n > 0:
            h[0] = 1.0
            if n % 2 == 0:
                h[n // 2] = 1.0
                h[1:n // 2] = 2.0
            else:
                h[1:(n + 1) // 2] = 2.0
        analytic = torch.fft.ifft(xf * h, dim=2)           # complex, (B,C,pad_len)

        # ---- frequency shift (reference: shift_func = exp(2j*pi*f_shift*dt*t)) ----
        t_axis = torch.arange(pad_len, device=x.device, dtype=x.dtype)
        shift_func = torch.exp(2j * torch.pi * f_shift * self.dt * t_axis)
        shifted = analytic * shift_func                     # (B,C,pad_len)

        return shifted[:, :, :t].real.to(x.dtype)

    def __repr__(self) -> str:
        return f"FrequencyShift(freq={self.freq_mod}, dt={self.dt}, p={self.p})"


# ---- select-channel variants (reference: random_noise, data_flipping, data_scale) ----

class ChannelNoise:
    """Add uniform noise to *selected* EEG channels.

    Reference: ``random_noise`` (line 19-35)
      channels_selected = random.choice(all_channels, channel_num)
      lam = uniform(-0.5, 0.5)
      noise = lam * data[:,selected,:].std(axis=-1) / C_noise   ← per-channel std
      noisy = copy → add noise on selected channels

    Differs from ``UniformNoiseTrial`` in two ways:
      (a) only a subset of channels are perturbed,
      (b) noise magnitude uses per-channel std, not per-trial std.

    Parameters
    ----------
    C_noise     : float = 2.0   (divisor, larger → weaker)
    channel_num : int   = 1     (how many channels to perturb)
    p           : float = 0.5
    """

    def __init__(self, C_noise: float = 2.0, channel_num: int = 1, p: float = 0.5):
        self.C_noise = C_noise
        self.channel_num = channel_num
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        batch, chans, _ = x.shape
        # reference: lam = np.random.uniform(-0.5, 0.5, size=1)
        lam = torch.empty(batch, 1, 1, device=x.device).uniform_(-0.5, 0.5)
        out = x.clone()
        for b in range(batch):
            k = min(self.channel_num, chans)
            selected = torch.randperm(chans, device=x.device)[:k]
            # reference: noise = lam * noise.std(axis=-1) / C_noise
            noise = lam[b] * x[b, selected, :].std(dim=1, keepdim=True) / self.C_noise
            out[b, selected, :] = out[b, selected, :] + noise
        return out

    def __repr__(self) -> str:
        return f"ChannelNoise(C={self.C_noise}, ch={self.channel_num}, p={self.p})"


class ChannelFlip:
    """Flip selected channels vertically:  max - data.

    Reference: ``data_flipping`` (line 38-48)
      max_values = max(data, axis=-1, keepdims=True)
      flipped[:, selected, :] = max_values[:, selected, :] - data[:, selected, :]

    Parameters
    ----------
    channel_num : int = 1
    p           : float = 0.5
    """

    def __init__(self, channel_num: int = 1, p: float = 0.5):
        self.channel_num = channel_num
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        batch, chans, _ = x.shape
        out = x.clone()
        for b in range(batch):
            k = min(self.channel_num, chans)
            selected = torch.randperm(chans, device=x.device)[:k]
            # reference: max_values = np.max(data, axis=-1, keepdims=True)
            mx = x[b, selected, :].amax(dim=1, keepdim=True)
            out[b, selected, :] = mx - x[b, selected, :]
        return out

    def __repr__(self) -> str:
        return f"ChannelFlip(ch={self.channel_num}, p={self.p})"


class ChannelScale:
    """Scale selected channels by (1 ± multi).

    Reference: ``data_scale`` (line 51-65)
      if strengthen: pre_data[:,selected,:] = data[:,selected,:] * (1 + multi)
      if weaken:     pre_data[:,selected,:] = data[:,selected,:] * (1 - multi)

    Parameters
    ----------
    multi       : float = 0.05
    channel_num : int   = 1
    p           : float = 0.5
    """

    def __init__(self, multi: float = 0.05, channel_num: int = 1, p: float = 0.5):
        self.multi = multi
        self.channel_num = channel_num
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        batch, chans, _ = x.shape
        # reference randomly picks strengthen or weaken
        strengthen = torch.rand(1).item() > 0.5
        factor = 1.0 + self.multi if strengthen else 1.0 - self.multi
        out = x.clone()
        for b in range(batch):
            k = min(self.channel_num, chans)
            selected = torch.randperm(chans, device=x.device)[:k]
            out[b, selected, :] = out[b, selected, :] * factor
        return out

    def __repr__(self) -> str:
        return f"ChannelScale(multi={self.multi}, ch={self.channel_num}, p={self.p})"


class TimeReverse:
    """Flip the signal along the time axis.

    Reference: ``channel_reverse`` (line 533-550)
      reversed_sample = np.flip(data[i], axis=-1)

    Parameters
    ----------
    p : float = 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        # reference: np.flip along axis=-1
        return torch.flip(x, dims=[2])

    def __repr__(self) -> str:
        return f"TimeReverse(p={self.p})"


# ======================================================================
# B. Pair-based augmentations — mix two same-class trials
#    Reference generates all pairs offline and concatenates.
#    Online equivalent: for each sample, with prob p, fetch a second
#    same-class sample from the dataset and mix.
# ======================================================================

class ChannelMixup:
    """Linear interpolation of two same-class trials:  p*s1 + (1-p)*s2.

    Reference: ``channel_mixup`` (line 468-499)
      p = random.uniform(0.3, 0.7)
      new_sample = p * sample1 + (1-p) * sample2

    The second sample is supplied by the caller (dataset).
    """

    def __init__(self, mix_lo: float = 0.3, mix_hi: float = 0.7, p: float = 0.5):
        self.mix_lo = mix_lo
        self.mix_hi = mix_hi
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # reference: p = np.random.uniform(0.3, 0.7)
        alpha = torch.empty(1, device=x1.device).uniform_(self.mix_lo, self.mix_hi).item()
        return alpha * x1 + (1.0 - alpha) * x2

    def __repr__(self) -> str:
        return f"ChannelMixup(alpha=[{self.mix_lo},{self.mix_hi}], p={self.p})"


class TrialMixup:
    """Swap the first half of two trials (time-domain splice).

    Reference: ``trial_mixup`` (line 501-531)
      swap sample1[:,:T//2] with sample2[:,:T//2]
      keep both as two new augmented samples

    The second sample is supplied by the caller (dataset).

    Parameters
    ----------
    p : float = 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # reference: swap first half along time axis
        t_half = x1.shape[2] // 2
        out1, out2 = x1.clone(), x2.clone()
        out1[:, :, :t_half], out2[:, :, :t_half] = x2[:, :, :t_half], x1[:, :, :t_half]
        return out1, out2

    def __repr__(self) -> str:
        return f"TrialMixup(p={self.p})"


class ChannelMixure:
    """Replace a few channels with data from another *same-class* trial.

    Reference: ``channel_mixure`` (line 446-465)
      for each augmented sample:
        select_ch   = random C channels
        select_idx  = random N trials (same class)
        new_trial[select_ch[ch], :] = reference_trial[select_idx[ch], select_ch[ch], :]

    The second sample is supplied by the caller (dataset).

    Parameters
    ----------
    channel_num : int = 5   (how many channels to swap)
    p           : float = 0.5
    """

    def __init__(self, channel_num: int = 5, p: float = 0.5):
        self.channel_num = channel_num
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        chans = x1.shape[1]
        k = min(self.channel_num, chans)
        out = x1.clone()
        for i in range(k):
            # reference: random channel + random trial for each swap
            ch = torch.randint(0, chans, (1,), device=x1.device).item()
            out[:, ch, :] = x2[:, ch, :]
        return out

    def __repr__(self) -> str:
        return f"ChannelMixure(ch={self.channel_num}, p={self.p})"


class DWTA:
    """Discrete Wavelet Transform augmentation — swap detail coefficients.

    Reference: ``DWTA`` (line 663-669) + ``use_DWTA`` (line 673-711)
      wavelet 'db5':  TcA, TcD = dwt(target);
      swap:  aug1 = idwt(src_A, target_D)   /  aug2 = idwt(target_A, src_D)

    NOTE: requires ``pywt`` (PyWavelets).  Falls back to identity if not installed.
    The second sample is supplied by the caller (dataset).
    """

    def __init__(self, wavelet: str = "db5", p: float = 0.5):
        self.wavelet = wavelet
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        try:
            import pywt
        except ImportError:
            return x1  # pywt not available, skip

        batch, chans, t = x1.shape
        x1_np = x1.cpu().numpy()
        x2_np = x2.cpu().numpy()
        out = np.empty_like(x1_np)

        for b in range(batch):
            for c in range(chans):
                # reference: TcA, TcD = pywt.dwt(target, wavename)
                cA1, cD1 = pywt.dwt(x1_np[b, c, :], self.wavelet)
                cA2, cD2 = pywt.dwt(x2_np[b, c, :], self.wavelet)
                # reference: swap detail coeffs from target, approx from source
                rec = pywt.idwt(cA1, cD2, self.wavelet, 'smooth')
                # trim / pad to original length (pywt may change length)
                if len(rec) > t:
                    rec = rec[:t]
                elif len(rec) < t:
                    rec = np.pad(rec, (0, t - len(rec)))
                out[b, c, :] = rec

        return torch.from_numpy(out).to(x1.device, dtype=x1.dtype)

    def __repr__(self) -> str:
        return f"DWTA(wavelet={self.wavelet}, p={self.p})"


# ======================================================================
# C. Channel-mirror augmentation — needs EEG channel-name map
# ======================================================================

class MirrorReverse:
    """Swap left↔right hemisphere channel pairs based on channel names.

    Reference: ``mirror_reverse`` (line 94-118)
      Pairs are formed by matching odd/even channel numbers within each
      scalp region (Fp, AF, F, FC, FT, C, T, CP, TP, P, PO, O).
      e.g. C3 ↔ C4, FC1 ↔ FC2.

    Default channel names are for BNCI2014001 (BCI Comp. IV-2a, 22 ch):
      Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6,
      CP3, CP1, CPz, CP2, CP4, P1, Pz, P2, POz
    Mirror pairs: FC3↔FC4, FC1↔FC2, C5↔C6, C3↔C4, C1↔C2,
                  CP3↔CP4, CP1↔CP2, P1↔P2  (8 pairs).

    Also flips left_hand ↔ right_hand labels.

    Parameters
    ----------
    ch_names : list[str] | None
        22 channel names; defaults to BNCI2014001 montage.
    p : float = 0.5
    """

    # BNCI2014001 (BCI Competition IV-2a) 22-channel montage
    _BNCI2014001_CHANNELS = [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz",
    ]

    def __init__(self, ch_names: list[str] | None = None, p: float = 0.5):
        if ch_names is None:
            ch_names = self._BNCI2014001_CHANNELS
        self.ch_names = ch_names
        self.p = p
        self._pair_indices = self._build_pairs(ch_names)

    @classmethod
    def _build_pairs(cls, ch_names: list[str]) -> list[tuple[int, int]]:
        """Build canonical left↔right channel pairs: odd N ↔ odd N+1.

        Standard EEG mirror pairs (same scalp region, consecutive numbers):
          FC3↔FC4, FC1↔FC2, C5↔C6, C3↔C4, C1↔C2,
          CP3↔CP4, CP1↔CP2, P1↔P2.
        """
        import re
        # Index channels by (prefix, number)
        indexed: dict[tuple[str, int], int] = {}
        for i, name in enumerate(ch_names):
            m = re.match(r"^([A-Za-z]+)(\d+)$", name)
            if m:
                indexed[(m.group(1), int(m.group(2)))] = i

        pairs: list[tuple[int, int]] = []
        for (prefix, num), i in indexed.items():
            if num % 2 == 0:
                continue  # only start from odd (left hemisphere)
            # Look for corresponding even channel: same prefix, num+1
            j = indexed.get((prefix, num + 1))
            if j is not None:
                pairs.append((i, j))
        return pairs

    def swap_channels(self, x: torch.Tensor) -> torch.Tensor:
        """Swap left↔right channels according to the pair map."""
        if not self._pair_indices:
            return x
        out = x.clone()
        for i, j in self._pair_indices:
            out[:, j, :], out[:, i, :] = x[:, i, :].clone(), x[:, j, :].clone()
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        return self.swap_channels(x)

    def __repr__(self) -> str:
        return f"MirrorReverse(pairs={len(self._pair_indices)}, p={self.p})"


# ======================================================================
# D. Multi-noise wrapper — applies one of several noise types
# ======================================================================

class MultiNoise:
    """Apply one of several noise types (gaussian / salt&pepper / Poisson / pink).

    Reference: ``channel_noise`` (line 552-624)
      Supports four noise types selectable by name.

    Parameters
    ----------
    noise_type : str  = "gaussian"  (one of gaussian, salt_and_pepper, poisson, pink)
    p          : float = 0.5
    """

    def __init__(self, noise_type: str = "gaussian", p: float = 0.5):
        if noise_type not in ("gaussian", "salt_and_pepper", "poisson", "pink"):
            raise ValueError(f"Unknown noise_type: {noise_type}")
        self.noise_type = noise_type
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not self.p or torch.rand(1).item() > self.p:
            return x
        if self.noise_type == "gaussian":
            return self._gaussian(x)
        elif self.noise_type == "salt_and_pepper":
            return self._salt_pepper(x)
        elif self.noise_type == "poisson":
            return self._poisson(x)
        else:
            return self._pink(x)

    # ---- individual noise generators (matching reference line 563-598) ----

    @staticmethod
    def _gaussian(x: torch.Tensor) -> torch.Tensor:
        # reference: noise = np.random.normal(mean=0.0, std=0.1, data.shape)
        return x + torch.randn_like(x) * 0.1

    @staticmethod
    def _salt_pepper(x: torch.Tensor) -> torch.Tensor:
        # reference: randomly set pixels to max or min with probability salt_prob/pepper_prob
        out = x.clone()
        prob = 0.01
        # salt (set to max)
        salt_mask = torch.rand_like(x) < prob
        out[salt_mask] = x.max()
        # pepper (set to min)
        pepper_mask = torch.rand_like(x) < prob
        out[pepper_mask] = x.min()
        return out

    @staticmethod
    def _poisson(x: torch.Tensor) -> torch.Tensor:
        # reference: noise = np.random.poisson(size=data.shape)
        x_np = x.cpu().numpy()
        noise = np.random.poisson(size=x_np.shape).astype(x_np.dtype)
        return x + torch.from_numpy(noise).to(x.device)

    @staticmethod
    def _pink(x: torch.Tensor) -> torch.Tensor:
        # reference: Voss-McCartney pink noise (line 583-598)
        batch, chans, t = x.shape
        num_cols = int(np.ceil(np.log2(t)))
        pad_len = 2 ** num_cols
        # generate pink noise for each (b,c) via Voss-McCartney
        noise = np.zeros((batch, chans, pad_len), dtype=np.float32)
        white = np.random.randn(batch, chans, pad_len).astype(np.float32)
        for i in range(1, num_cols):
            noise[:, :, ::2 ** i] += white[:, :, ::2 ** i]
        noise = noise[:, :, :t]
        # scale by frequency decay
        alpha = 1.0
        scale = (np.arange(t) + 1) ** (-alpha / 2.0)
        noise = (noise * scale.reshape(1, 1, -1)).astype(np.float32)
        return x + torch.from_numpy(noise).to(x.device, dtype=x.dtype)

    def __repr__(self) -> str:
        return f"MultiNoise(type={self.noise_type}, p={self.p})"


# ======================================================================
# E. Pipeline
# ======================================================================

_BUILTIN: dict[str, type] = {
    # trial-level (all channels) — from data_noise_f / data_mult_f / data_neg_f / freq_mod_f
    "noise_trial": UniformNoiseTrial,
    "mult_trial":  AmplitudeMultTrial,
    "neg":         AmplitudeNeg,
    "freq_shift":  FrequencyShift,
    # select-channel variants — from random_noise / data_flipping / data_scale
    "noise_ch":    ChannelNoise,
    "flip_ch":     ChannelFlip,
    "scale_ch":    ChannelScale,
    # time reversal — from channel_reverse
    "time_reverse": TimeReverse,
    # mirror — from mirror_reverse / augment_with_CR
    "mirror":      MirrorReverse,
    # multi-noise types — from channel_noise (one key per noise type)
    "noise_gaussian": lambda p=0.5: MultiNoise("gaussian", p=p),
    "noise_sp":       lambda p=0.5: MultiNoise("salt_and_pepper", p=p),
    "noise_poisson":  lambda p=0.5: MultiNoise("poisson", p=p),
    "noise_pink":     lambda p=0.5: MultiNoise("pink", p=p),
}

# Pair-based augmentations — handled separately by AugmentedDataset
# because they need a second same-class sample from the dataset.
_PAIR_BUILTIN: dict[str, type] = {
    "mixup":       ChannelMixup,
    "ch_mixure":   ChannelMixure,
    "trial_mixup": TrialMixup,
    "dwta":        DWTA,
}


class AugmentationPipeline:
    """Apply a sequence of single-sample augmentations to a batch.

    Parameters
    ----------
    names : str | list[str]
        ``"all"`` for every single-sample method, or a list of keys from
        ``_BUILTIN``.
    mirror_ch_names : list[str] | None
        Channel names for mirror-reverse; only required when ``"mirror"`` is in
        the pipeline.
    p : float = 0.5
        Shared application probability for every method.
    """

    def __init__(self, names: str | list[str] = "all", mirror_ch_names: list[str] | None = None, p: float = 0.5):
        if names == "all" or names == ["all"]:
            keys = list(_BUILTIN)
        else:
            keys = [n for n in names if n in _BUILTIN]
            unknown = set(names) - set(_BUILTIN)
            if unknown:
                print(f"Warning: unknown augmentation keys ignored: {unknown}")
        self.augmentations: list = []
        for k in keys:
            if k == "mirror":
                self.augmentations.append(MirrorReverse(ch_names=mirror_ch_names, p=p))
            else:
                self.augmentations.append(_BUILTIN[k](p=p))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x)
        return x

    def __repr__(self) -> str:
        names = [a.__class__.__name__ for a in self.augmentations]
        return f"AugmentationPipeline({names})"
