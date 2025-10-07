import matplotlib.pyplot as plt
import numpy as np

class PlotUtil:
    def __init__(self):
        pass

    def pick_2d(self, spec_like):
        """Accept [array], array, (1,F,T), (F,T,1), (C,F,T), etc. → return 2D (F, T)."""
        arr = spec_like[0] if isinstance(spec_like, (list, tuple)) else np.asarray(spec_like)
        # Squeeze singleton dims
        if arr.ndim >= 3 and 1 in arr.shape:
            arr = np.squeeze(arr)
        if arr.ndim == 3:         # pick first channel if still 3D
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected a 2D spectrogram after squeezing, got {arr.shape}")
        # Heuristic: freq bins are usually fewer than time frames
        F, T = arr.shape
        if F > T:
            arr = arr.T  # ensure (F, T)
        return arr  # (F, T), linear PSD

    def power_to_db(self, power, ref=1.0, floor_db=-100.0, eps=1e-18):
        """Linear PSD → dB re `ref` (default 1 unit²/Hz)."""
        db = 10.0 * np.log10(np.maximum(power, eps) / ref)
        return np.maximum(db, floor_db)

    def unit_label(self, unit: str, scale: str):
        # For colorbar label
        if scale == "linear":
            return f"{unit}^2/Hz"
        # dB
        return f"dB re 1 {unit}^2/Hz"

    def plot_vibration_spectrogram(
        self,
        spec_like,
        *,
        fs=200,           # sampling rate [Hz] → for axis in seconds/Hz (optional)
        hop=64,          # hop size [samples] (optional)
        nperseg=256,      # window length → for freq bins (optional; improves extent)
        unit="g",          # "g" or "m/s^2" (use whatever your PSD unit is)
        scale="dB",        # "dB" or "linear"
        db_floor=-5.0,
        title=None,
        fmax=None          # optionally cap visible frequency (Hz)
    ):
        S = self.pick_2d(spec_like)  # (F, T), linear PSD in unit^2/Hz
        F, T = S.shape

        # Convert to dB if requested
        if scale.lower() == "db":
            S_plot = self.power_to_db(S, ref=1.0, floor_db=db_floor)
        else:
            S_plot = S

        # Build axes if fs/hop provided; else fall back to frames/bins
        extent = None
        if fs is not None and hop is not None:
            # Frequency vector (best with nperseg; else infer from F)
            if nperseg is None:
                # F should equal nfft//2 + 1 → infer nperseg roughly
                nperseg = (F - 1) * 2
            freqs = np.fft.rfftfreq(nperseg, d=1.0/fs)
            # Map vertical bins to these centers
            # If freqs length mismatches slightly, just limit to min length
            m = min(len(freqs), F)
            freqs = freqs[:m]
            S_plot = S_plot[:m, :]
            # Time axis from hop
            times = np.arange(T) * (hop / fs)
            extent = [times[0], times[-1] + hop / fs, freqs[0], freqs[-1]]

        # Apply frequency cap for visualization if requested
        if fmax is not None and extent is not None:
            # Compute row index for fmax
            f0, f1 = extent[2], extent[3]
            if fmax < f1:
                frac = (fmax - f0) / (f1 - f0)
                rows = int(max(1, np.floor(frac * S_plot.shape[0])))
                S_plot = S_plot[:rows, :]
                extent[3] = fmax

        # Titles and labels
        if title is None:
            base = "Acceleration PSD Spectrogram"
            title = base + (" (dB)" if scale.lower() == "db" else " (linear)")

        plt.figure(figsize=(8, 4))
        im = plt.imshow(
            S_plot,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=extent  # None → frames/bins; else seconds/Hz
        )
        cbar = plt.colorbar(im)
        cbar.set_label(self.unit_label(unit, scale=scale.lower()))
        plt.xlabel("Time (s)" if extent is not None else "Time (frames)")
        plt.ylabel("Frequency (Hz)" if extent is not None else "Frequency (bins)")
        plt.title(title)
        plt.tight_layout()
        plt.show()