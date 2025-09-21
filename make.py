import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming
from pathlib import Path

sampling_frequency = 12800  # in Hz
sampling_period = 1 / sampling_frequency  # in seconds

def save(directory_path: str, target_path: str, clazz : str):
    ds = os.listdir(directory_path)
    
    for f in ds:

        if (("Session" in f and "_rel_time" in f)):
            df = pd.DataFrame(pd.read_csv(str(directory_path + f)))
            falls = prep_df(df, ["sensor_1", "sensor_2", "sensor_3", "sensor_4"], time_dff=3.0)
            no_falls = find_none_falls(df, ["sensor_1", "sensor_2", "sensor_3", "sensor_4"], time_dff=3.0)


            for fall in falls: 

                spec, labels = create_spectrograms(fall, clazz, train_seconds=[(fall["time_sec"].min(), fall["time_sec"].max())], nperseg=512)
                print(f)
                store_spectograms(spec, target_path, f.split(".csv")[0])
                #save_labels(labels, Path(target_path), "labels.npy")
            
            for no_fall in no_falls:
                spec, labels = create_spectrograms(no_fall, "No Fall", train_seconds=[(no_fall["time_sec"].min(), no_fall["time_sec"].max())], nperseg=512)
                store_spectograms(spec, './.spec/train/NoFall/', f.split(".csv")[0])

            
        elif './.data/test' in directory_path:
            df = pd.DataFrame(pd.read_csv(str(directory_path + f)))
            falls = prep_df(df, ["sensor_1", "sensor_2", "sensor_3", "sensor_4"], time_dff=3.0)

            for fall in falls: 

                spec, labels = create_spectrograms(fall, clazz, train_seconds=[(fall["time_sec"].min(), fall["time_sec"].max())], nperseg=512)
                print(f)
                store_spectograms(spec, target_path, f.split(".csv")[0])
        elif "False_labeled" in f:
            df = pd.DataFrame(pd.read_csv(str(directory_path + f)))
            falls = pred_no_fall_df(df, ["sensor_1", "sensor_2", "sensor_3", "sensor_4"], time_dif=3.0)
            
            for fall in falls:
                spec, labels = create_spectrograms(fall, clazz, train_seconds=[(fall["time_sec"].min(), fall["time_sec"].max())], nperseg=512)
                print(f)
                store_spectograms(spec, target_path, f.split(".csv")[0])

def find_none_falls(df: pd.DataFrame, sensors: list[str], time_dff : float = 5.0, time_col : str = 'time_sec') -> list:
    result = []

    for sensor in sensors:
        event_peak = df[sensor].max()

        sensor_peak_time = float(df[df[sensor] == event_peak][time_col].max())
        start_off_fall = float(sensor_peak_time) - time_dff
        end_off_fall = float(sensor_peak_time) + time_dff

        non_fall_df = df[(df[time_col] < start_off_fall) | (df[time_col] > end_off_fall)]
        result.append(non_fall_df)

    return result



def prep_df(df : pd.DataFrame, sensors : list[str], time_dff: float = 5.0, time_col : str = 'time_sec') -> list:

    result = []

    for sensor in sensors:
        event_peak = df[sensor].max()

        sensor_peak_time = float(df[df[sensor] == event_peak][time_col].max())
        
        start_time_threshold = float(sensor_peak_time) - time_dff
        
        fall_start_time = df.loc[df[time_col] >= start_time_threshold, time_col].min()

        end_time_threshold = float(sensor_peak_time) + time_dff

        fall_end_time = df.loc[df[time_col] <= end_time_threshold, time_col].max()

        fall_df = df[(df[time_col] >= fall_start_time) & (df[time_col] <= fall_end_time)]
        result.append(fall_df)

    return result

def pred_no_fall_df(df: pd.DataFrame, sensors : list [str], time_dif: float = 5.0, time_col : str = 'time_sec') -> list[pd.DataFrame]:
    result =  []
    
    start_time = 0.0
    for sensor in sensors:
        
        current_treshold = start_time + time_dif
        current_df = df[(df[time_col] >= start_time) & (df[time_col] < current_treshold)]
        result.append(current_df)

        start_time += time_dif
    
    return result


def make_spectograms(df : pd.DataFrame, 
                     sensor_name : str, 
                     file_name : str, 
                     nperseg = 512, 
                     sampling_frequency = 12800, 
                     bin_width = 2.0,
                     closed="right"):
    # TODO Fix so that we are createing spectograms correctly
    
    if isinstance(sensor_name, tuple) and len(sensor_name) == 1:
        sensor_name = sensor_name[0]
    if isinstance(sensor_name, (list, tuple)):
        raise ValueError("Pass a single sensor column name (string), not a list/tuple.")
    
    
    dataFrames : list[pd.DataFrame] = []
    
    for i in range(round(df["time_sec"].min()), round(df["time_sec"].max())):
    
        start = i # start time in seconds
        end = int(i + bin_width)

        edges = np.arange(start, end + 1, 1)

        cuts = pd.cut(df["time_sec"], bins=edges, right=True, include_lowest=True)

        dfs_by_second = df.assign(second_bin=cuts)

        dataFrames.append(dfs_by_second)


    for i in dataFrames:
        #if column.startswith("gearbox_vibration"): # This can be used to filter out specific columns            
            data = i[sensor_name]
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            plt.specgram(data)
            plt.savefig(file_name + sensor_name + '.png', bbox_inches='tight', pad_inches=0)
            plt.close()
    
def create_spectrograms(df : pd.DataFrame, fault_category : str, SECONDS : int=2, nperseg=None, train_seconds=None, val_seconds=None, test_seconds=None, mode='train'):
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

    if mode == 'train':
        segments = train_seconds or []
    elif mode == 'val':
        segments = val_seconds or []
    elif mode == 'test':
        segments = test_seconds or []
    else:
        raise ValueError("mode must be 'train', 'val' or 'test'")

    samples_per_interval = int(sampling_frequency * SECONDS)

    for start_sec, end_sec in segments:
        print(end_sec)
        print(start_sec)
        num_segments = int((end_sec - start_sec) // SECONDS)

        for i in range(num_segments):
            segment_start_time = start_sec + i * SECONDS
            segment_end_time = segment_start_time + SECONDS

            start_sample = int(segment_start_time * sampling_frequency)
            end_sample = start_sample + samples_per_interval

            spec_8ch = get_8_channel_spectrogram(
                df[(df["time_sec"] >= segment_start_time) & (df["time_sec"] <= segment_end_time)], sampling_frequency,
                segment_start_time, segment_end_time, nperseg)

            all_spectrograms.append(spec_8ch)
            all_labels.append(fault_category)

        print(f"Generated {i+1} spectrograms - label: {fault_category}")

    return all_spectrograms, all_labels

# This function turns all 8 sensor streams into spectrograms. Later we choose to use only the vibration sensors from the gearbox. 
def get_8_channel_spectrogram(data, sampling_frequency, base_name, start_time, end_time, nperseg = 512):

    all_spectrograms = []
    all_labels = []

    # Slice to first 5 seconds if desired
    
    for column in data:
        if column.startswith("sensor_1") or column.startswith("sensor_2") or column.startswith("sensor_3") or column.startswith("sensor_4"): # This can be used to filter out specific columns
            col = data[column]

            if len(col) < 256:
                all_spectrograms.append([])
                break

            w = hamming(nperseg) # Hamming window
            Sft = ShortTimeFFT(w, hop=int(nperseg*0.25), fs=sampling_frequency, scale_to='psd')
            Sxx = Sft.spectrogram(col.values)  # calculate absolute square of STFT
            all_spectrograms.append(Sxx)
    spec_8ch = np.stack(all_spectrograms, axis=0)

    return spec_8ch

def store_spectograms (spectrograms, path, event):
    
    index = 0
    for spectrogram in spectrograms:
        np.save(path + event + str(index) + ".npy", spectrogram)
        index += 1

def load(directory: str) -> list[np.ndarray]:
    files = os.listdir(directory)
    
    result = []

    for file in files:
        arr = np.load(directory + "/" + file)
        result.append(arr)

    return result

def save_labels(labels: list, storeage_path:Path, file_name: str):
    np.save(storeage_path/file_name, labels)