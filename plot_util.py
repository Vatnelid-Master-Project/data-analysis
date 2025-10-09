import matplotlib.pyplot as plt
import numpy as np

class PlotUtil:
    def __init__(self, spectrograms):
        self.spectrograms = spectrograms        

    def show_spec(self, index, title):
        hop = int(256 * 0.25)
        frame_times = np.arange(self.spectrograms[index].shape[1]) * hop / 200
        plt.figure(figsize=(8,4))

        plt.imshow(self.spectrograms[index], origin="lower", aspect="auto", 
                extent=[frame_times[0], frame_times[-1], 0, 200])
        plt.xlabel("Time (S)")
        plt.ylabel("Frequency bins")
        plt.title(title)
        plt.colorbar(label="amplitude in decibel scale")
        plt.show()