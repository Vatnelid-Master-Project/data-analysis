import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from pathlib import Path

sampling_frequency = 200 # Hz

def create_false_spectrogams(df: pd.DataFrame, SECONDS: int = 10, nperseg=None, train_seconds=None, val_seconds=None, test_seconds=None, mode='train'):
    all_spectrograms = []
    all_labels = []
    segments = []

    treshold = 5

    peak_time = df[df["sensor_1"] == df["sensor_1"].max()]["time_sec"].max()

    start_fall = peak_time - treshold
    end_fall = peak_time + treshold
    
    segments.append((df["time_sec"].min(), start_fall))
    segments.append((end_fall, df["time_sec"].max()))

    for start_sec, end_sec in segments:
        num_segments = int((end_sec - start_sec) // SECONDS)

        

        for i in range(num_segments):
            segment_start_time = start_sec + i * SECONDS
            segment_end_time = segment_start_time + SECONDS

            spec_8ch = get_8_channel_spectrogram(
                df[(df["time_sec"] >= segment_start_time) & (df["time_sec"] <= segment_end_time)], nperseg)

            all_spectrograms += spec_8ch

            all_labels.append("Not Fall")
            all_labels.append("Not Fall")
            all_labels.append("Not Fall")
            all_labels.append("Not Fall")
            
            print(f"Generated {i+1} spectrograms - label: Not Fall")

    return all_spectrograms, all_labels


def create_true_spectrograms(df : pd.DataFrame, SECONDS : int=2, nperseg=None, train_seconds=None, val_seconds=None, test_seconds=None, mode='train'):
    """
    Generates and returns spectrograms for specified time segments.

    Parameters:
    dataDict (dict): Dict containing fault category data.
    fault_category (str): Category to process.
    SECONDS (int): Length of each spectrogram segment in seconds.
    nperseg (int): FFT window size.
    train_seconds (list of tuples): List of (start, end) in seconds for training.
    val_seconds (list of tuples): List of (start, end) in seconds for validation.
    mode (str): 'train' or 'val' - determines which segments to use.

    Returns:
    spectrograms, labels
    """
    all_spectrograms = []
    all_labels = []
    segments = []

    treshold = 4

    peak_time = df[df["sensor_1"] == df["sensor_1"].max()]["time_sec"].max()

    start_time = peak_time - treshold
    end_time = peak_time + treshold
    
    segments.append((start_time, end_time))

    for start_sec, end_sec in segments:

        spec_8ch = get_8_channel_spectrogram(
            df[(df["time_sec"] >= start_time) & (df["time_sec"] <= end_time)], nperseg)

        all_spectrograms += spec_8ch
        all_labels.append("Fall")
        all_labels.append("Fall")
        all_labels.append("Fall")
        all_labels.append("Fall")

        print(f"Generated {1} spectrograms - label: Fall")

    return all_spectrograms, all_labels

def create_fall_spectrograms(df : pd.DataFrame, fault_category : str, SECONDS : int=2, nperseg=None, train_seconds=None, val_seconds=None, test_seconds=None, mode='train'):
    """
    Generates and returns spectrograms for specified time segments.

    Parameters:
    dataDict (dict): Dict containing fault category data.
    fault_category (str): Category to process.
    SECONDS (int): Length of each spectrogram segment in seconds.
    nperseg (int): FFT window size.
    train_seconds (list of tuples): List of (start, end) in seconds for training.            all_labels.append(fault_category)
    val_seconds (list of tuples): List of (start, end) in seconds for validation.
    mode (str): 'train' or 'val' - determines which segments to use.

    Returns:
    spectrograms, labels
    """
    all_spectrograms = []
    all_labels = []
    segments = []

    treshold = 5

    peak_time = df[df["sensor_1"] == df["sensor_1"].max()]["time_sec"].max()

    start_time = peak_time - treshold
    end_time = peak_time + treshold
    
    segments.append((start_time, end_time))

    for start_sec, end_sec in segments:

        spec_8ch = get_8_channel_spectrogram(
            df[(df["time_sec"] >= start_time) & (df["time_sec"] <= end_time)], nperseg)

        all_spectrograms += spec_8ch

        # One fault category for each sensor (the fault is the same)
        all_labels.append(fault_category)
        all_labels.append(fault_category)
        all_labels.append(fault_category)
        all_labels.append(fault_category)


        print(f"Generated {1} spectrograms - label: {fault_category}")

    return all_spectrograms, all_labels

def get_8_channel_spectrogram(data, sampling_frequency = 200, nperseg = 256, dynamic_range_db = 60, band_max_hz = 200):

    all_spectrograms = []

    # Slice to first 5 seconds if desired
    
    for column in data:
        if column.startswith("sensor_1") or column.startswith("sensor_2") or column.startswith("sensor_3") or column.startswith("sensor_4"): # This can be used to filter out specific columns
            
            col = data[column]
            w = hamming(nperseg) # Hamming window
            Sft = ShortTimeFFT(w, hop=int(nperseg*0.25), fs=sampling_frequency, scale_to='psd')
            Sxx = Sft.spectrogram(col.values)  # calculate absolute square of STFT

            # --- convert to dB and clip dynamic range ---
            # Normalization - turn spectrogram into dB scale
            Sxx_db = 10*np.log10(Sxx + 1e-12)
            vmax = np.percentile(Sxx_db, 95)
            vmin = vmax - dynamic_range_db
            Sxx_db = np.clip(Sxx_db, vmin, vmax)

            # --- optional band limit on frequency axis ---
            f = np.linspace(0, sampling_frequency/2, Sxx_db.shape[0])
            if band_max_hz is not None:
                keep = f <= band_max_hz
                Sxx_db = Sxx_db[keep, :]

            all_spectrograms.append(Sxx_db.astype(np.float32))

    return all_spectrograms

def get_primary_data(dir: Path):
    directory = os.listdir(str(dir))
    
    data = []
    fault_labels = []

    for file in directory:
        if (("Session" in file and "_rel_time" in file)):
            df = pd.DataFrame(pd.read_csv(str(dir/file)))
            t_spec, t_labels = create_true_spectrograms(df, SECONDS=4, nperseg=256)
            f_spec, f_labels = create_false_spectrogams(df, SECONDS=8, nperseg=256)
            
            #Makes sure the length for none_falls is equal to the actuall falls.
            f_spec = f_spec[:len(t_spec)]
            f_labels = f_labels[:len(t_labels)]

            data += t_spec
            data += f_spec

            fault_labels += t_labels
            fault_labels += f_labels
    
    return data, fault_labels

def get_secondary_data(dir: Path, fault_category: str):
    directory = os.listdir(str(dir))
    
    data = []
    fault_labels = []

    for file in directory:
        if (("Session" in file and "_rel_time" in file)):
            df = pd.DataFrame(pd.read_csv(str(dir/file)))
            spec, labels = create_fall_spectrograms(df, fault_category, SECONDS=4, nperseg=256, train_seconds=[(df["time_sec"].min(), df["time_sec"].max())])
            data += spec
            fault_labels += labels
    
    return data, fault_labels

def load_primary_data():
    test_spectograms = []
    test_labels = []

    validation_path = Path('./.data/train')



    controlled = Path(validation_path/'controlledFall')
    hard = Path(validation_path/'hardFall')
    slipTrip = Path(validation_path/'Slip/Trip')

    
    controlled_falls, controlled_labels = get_primary_data(controlled)
    hard_falls, hard_labels = get_primary_data(hard)
    slip_trip_fall, slipTrip_labels = get_primary_data(slipTrip)
    
    test_spectograms += controlled_falls
    test_labels += controlled_labels

    test_spectograms += hard_falls
    test_labels += hard_labels

    test_spectograms += slip_trip_fall
    test_labels += slipTrip_labels

    return test_spectograms, test_labels

def load_secondary_data():
    test_spectograms = []
    test_labels = []

    validation_path = Path('./.data/train')



    controlled = Path(validation_path/'controlledFall')
    hard = Path(validation_path/'hardFall')
    slipTrip = Path(validation_path/'Slip/Trip')

    
    controlled_falls, controlled_labels = get_secondary_data(controlled, 'Controlled Fall')
    hard_falls, hard_labels = get_secondary_data(hard, 'Hard Fall')
    slip_trip_fall, slipTrip_labels = get_secondary_data(slipTrip, 'SlipTrip')
    
    test_spectograms += controlled_falls
    test_labels += controlled_labels

    test_spectograms += hard_falls
    test_labels += hard_labels

    test_spectograms += slip_trip_fall
    test_labels += slipTrip_labels

    return test_spectograms, test_labels