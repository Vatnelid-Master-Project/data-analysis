import matplotlib.pyplot as plt
import numpy as np

class PlotUtil:
    def __init__(self, spectrograms):
        self.spectrograms = spectrograms        

    def show_spec(self, index, title, fs=200, nperseg=64, band_max_hz=80):
        spec = self.spectrograms[index]          # shape = (n_freq_bins, n_frames)

        # 1) time axis (one x-step per STFT frame)
        hop = int(nperseg * 0.25)                # you used 25% hop
        n_frames = spec.shape[1]
        frame_times = np.arange(n_frames) * hop / fs   # seconds

        # 2) frequency axis (0 .. fs/2 mapped to rows)
        n_freq_bins = spec.shape[0]
        freqs = np.linspace(0, fs/2, n_freq_bins)

        # if you ever want to crop by band_max_hz:
        keep = freqs <= band_max_hz
        spec = spec[keep, :]
        freqs = freqs[keep]

        # 3) plot
        plt.figure(figsize=(8, 4))
        plt.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=[frame_times[0], frame_times[-1], freqs[0], freqs[-1]],
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