import matplotlib.pyplot as plt
import numpy as np

class PlotUtil:
    def __init__(self, spectrograms, seconds = 10):
        self.spectrograms = spectrograms
        self.seconds = seconds        

    def show_spec(self, index, title, fs=200, nperseg=64, band_max_hz=80):
        spec = self.spectrograms[index]          # shape = (n_freq_bins, n_frames)

        global_max = max(s.max() for s in spec)
        global_min = min(s.min() for s in spec)

        # 1) time axis (one x-step per STFT frame)
        # frequency axis from FFT bins
        n_freq_bins = spec.shape[0]
        freqs = np.linspace(0, fs/2, n_freq_bins)

        # crop frequency band
        keep = freqs <= band_max_hz
        spec = spec[keep, :]
        freqs = freqs[keep]

        # *** force x-axis to [0, self.seconds] ***
        t_min, t_max = 0.0, float(self.seconds)

        # 3) plot
        plt.figure(figsize=(8, 4))
        plt.imshow(
            spec,
            vmax=global_max,
            vmin=global_min,
            origin="lower",
            aspect="auto",
            extent=[t_min, t_max, freqs[0], freqs[-1]],
            interpolation="bilinear",          # just to make it look smoother
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()
    
    def show_full_fs_range(self, index, title, fs=200, nperseg=64):
        spec = self.spectrograms[index]          # shape = (n_freq_bins, n_frames)

        global_max = max(s.max() for s in spec)
        global_min = min(s.min() for s in spec)

        # 1) time axis (one x-step per STFT frame)
        # frequency axis from FFT bins

        # *** force x-axis to [0, self.seconds] ***
        t_min, t_max = 0.0, float(self.seconds)

        # 3) plot
        plt.figure(figsize=(8, 4))
        plt.imshow(
            spec,
            vmax=global_max,
            vmin=global_min,
            origin="lower",
            aspect="auto",
            extent=[t_min, t_max, 0, fs],
            interpolation="bilinear",          # just to make it look smoother
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()
    
    def plot_full_fall(self, index, times, freqs, title):
        spec = self.spectrograms[index]

        global_max = max(s.max() for s in spec)
        global_min = min(s.min() for s in spec)
        
        plt.figure(figsize=(10, 4))
        plt.imshow(
            spec,
            vmax=global_max,
            vmin=global_min,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="bilinear"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()
    
    def plot_nyquist_freq(self, index, title, fs=200, band_max_hz=80, duration=100):
        spec = self.spectrograms[index]          # shape = (n_freq_bins, n_frames)

        global_max = max(s.max() for s in spec)
        global_min = min(s.min() for s in spec)

        # 1) time axis (one x-step per STFT frame)
        # frequency axis from FFT bins
        n_freq_bins = spec.shape[0]
        freqs = np.linspace(0, fs/2, n_freq_bins)

        # crop frequency band
        keep = freqs <= band_max_hz
        spec = spec[keep, :]
        freqs = freqs[keep]

        # *** force x-axis to [0, self.seconds] ***
        t_min, t_max = 0.0, float(duration)

        # 3) plot
        plt.figure(figsize=(8, 4))
        plt.imshow(
            spec,
            vmax=global_max,
            vmin=global_min,
            origin="lower",
            aspect="auto",
            extent=[t_min, t_max, freqs[0], freqs[-1]],
            interpolation="bilinear",          # just to make it look smoother
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(title)
        plt.colorbar(label="Amplitude (dB)")
        plt.tight_layout()
        plt.show()


    
    @staticmethod
    def plot_tensor_image(img):
        image = img.detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))

        plt.imshow(image)
        plt.axis("off")
        plt.title("Tensor Image")
        plt.show()