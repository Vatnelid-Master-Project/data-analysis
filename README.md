# data-analysis
Uses CNN to classify types of falls from vibration data as spectrograms

# Set up

make sure to have pyhton 3.12 or newer

run the following commands:

create a new envrionment
`python -m venv .venv`

on linux / unix / macOS to source it

`source .venv/bin/activate`

on windows: 

`. .venv/bin/activate`

install depedencies

`pip install -r req.txt` 


# Calculations made step-by-step

First Step: Find the signal peak

- We must assume that the fall impact is were the signal peak is
    - For Falls
        - The session/segment is reduced to +- 5 seconds
        - Why 5 seconds exaclty?
            - This way we capture as much of movment this is related to the fall, while excluding as much of the non fall as possible
    - For Non Falls
        - The peak is found, but is removed from the segment, the remaining session is devided in to segments of 10 seconds each.

Step 2: The spectogram of each segment

- Hamming window
    - To reduce spectral leakage (Energy smearing across frequencies)
    - 64 samples, 100 samples = 500 ms
- ShortTimeFFT
    - What happens?
        - w defines the frame using the hamming window
        - hop: int (64*0,25) = 16, and with an overlap of 48 samples.
        - sampling frequency: explains it self, the freqency the signal is acquired at.
        - We move a little forward here, since we are using decibel normalization, we set scale to psd, which means that the scale in the spectogram is like a power spectral denisity rather the faw FFT magnitudes.
        - filter: Remove the low frequency content
        - Calculation:
        $$ S(f, t) = abs(X(f,t)^2
    - Sxx is 2D array:
        - rows = frequency bins
        - columns = time frames
        - values = vibration signals
- Decibel convertion
    - Since we are not using the ADC value, but the normalization after the spectral calculation, we use 
    - $$ 10*log10(s + 1e-12) 
    - (to avoid log of 0).
    - Normalization after spectral calculation is done to perserve the signal, use clipping, stablility for ML, and easier relative comparisons.
- Set the range:
    - 95 percentile is used to exclude the most extream values. If we had used the max, we could risk that we had a signal the was suddenly alot stronger than the others, making rest of the spectogram very dark and meaning less, while still perserving fall signals.
    - vmin set the lowest range, based on the max and the dynamic range. Anything below this is concidered irrelevant, noice ect.
    - Last is the clip, values above the max, is set to the max, values below min, is set to the min, and values within the range remains unchanged.

Step Three, prepare the spectogram for ML (Here is process the same for the models). This process is following the same steps as Maguns. 

- The models expect 1 channel images, so we need to assure it is one channel
- pyTourch expects the float64 type.
- since decibel scaling is already done, that is skipped.
- We want the peak of all the spectograms to be the same, so 0 db is maximum across all the spectograms.
- Normalize to [0,1]
- Applying gamma 0,9, brigthens darker values relative to bright ones to increase the visibility of faint structures.
- Since the ShortTimeFFT stores the frequency increasing downwards, the spectogram needs to be flipped for better visuals.
- Since the color map was commented before, magma is applied for nice red/orange colors, but does not have practical meaning to do.
- then, the numpy array is converted into a tensor.

- CNNs: Core building blocks
    - Conv2d
        - Learns:
            - edges, textures, shapes, freqency patterns
            - Deep Layers learn higher level features.
        - Used because images have spatial structure
    - Dropout: Disables naurons during the training
        - This reduces overfitting by preventing neurons from relying on each other
    - Relu: is the activation function that is returing back the signal containing the helpfull data, and null out the data that is not helpfull.
    - MaxPooling
        - Reduing the input size by returing the strongest signal.
        - reduces computation and memory.
        - Reducing overfitting.
    - BarchNorm
        - Normalizes layer output.
        - Stabilizes training.
        - Constantly re-centering and rescaling data so learning stays well-behaved.
    - Linear
        - Standard Matrix multiplication so that every input connects to every output.
        - This is the decision making layer.

- How does these work together when identifying cats (for instance)
    - Conv2d → detect whiskers, ears, shapes
    - Relu → Remove weak signals.
    - MaxPool → keep strongest detections
    - BatchNorm → keep signals stable
    - Dropout → avoid memorizing specific training cats
    - Linear → decide: “this is a cat”

Youdens’ J threshold: sensitivity + specificity.