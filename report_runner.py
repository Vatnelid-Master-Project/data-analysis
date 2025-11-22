from fastai.vision.all import *

from make import load_data_for_pipeline

class Report():
    def __init__(self, dataframe: pd.DataFrame, autoencoder: Learner, sd_model: Learner, ha_model: Learner, hybrid: Learner, classic: Learner):
        self.dataframe = dataframe
        self.autoencoder = autoencoder
        self.sd_model = sd_model
        self.ha_model = ha_model
        self.hybrid = hybrid
        self.classic = classic

    def run_autonencoder(self, img : TensorImage, threshold: float):
        mse = F.mse_loss(img, self.autoencoder.predict(img)[0])
        if (mse >= threshold):
            return "Fall"

        return "Not Fall" 

    def run_state_detection(self, img: TensorImage):
        return self.sd_model.predict(img)[0]

    def run_heath_assessment(self, img: TensorImage):
        return self.ha_model.predict(img)[0]

    def run_hybrid(self, img: TensorImage):
        return self.hybrid.predict(img)[0]

    def run_classic(self, img: TensorImage):
        return self.classic.predict(img)[0]

    def run_report(self, specs: list, true_labels: list):
        df = self.dataframe
        
        # Run autoencoder
        for i in range(len(specs)):
            auto_result = self.run_autonencoder(self.__get_tensor(specs[i]), 0.0781)
            sd_result = self.run_state_detection(self.__get_tensor(specs[i]))
            
            if auto_result == "Not Fall" and sd_result == "Not Fall":
                ha_result = "Not Fall"
                hybrid_result = "Not Fall"
                classic_result = "Not Fall"    
            else:
                ha_result = self.run_heath_assessment(self.__get_tensor(specs[i]))
                hybrid_result = self.run_hybrid(self.__get_tensor(specs[i]))
                classic_result = self.run_classic(self.__get_tensor(specs[i]))

            df.loc[len(df)] = [auto_result, sd_result, ha_result, hybrid_result, classic_result, true_labels[i]]
    
    def __get_tensor(self, arr):
        # If spectrogram is multi-channel, convert to grayscale first
        if len(arr.shape) == 3:
            arr = np.mean(arr, axis=-1)  # average across channels
        
        #arr_01 = arr_01[:, ::-1].copy() # flip spectrogram
        
        # Apply colormap to convert grayscale to color
        # Popular colormaps for spectrograms: 'viridis', 'plasma', 'magma', 'inferno', 'jet', 'hot'
        colormap = plt.get_cmap('hot')  # or try 'plasma', 'magma', 'jet'
        arr_colored = colormap(arr)  # This returns RGBA (H, W, 4)
        
        # Convert RGBA to RGB (drop alpha channel)
        arr_rgb = arr_colored[:, :, :3]  # (H, W, 3)
        
        # Convert to PyTorch tensor with correct channel ordering
        t = torch.tensor(arr_rgb).float().permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Resize image
        t_resized = torch.nn.functional.interpolate(t, size=(112, 112), mode='bilinear', align_corners=False)
        
        return TensorImage(t_resized.squeeze(0))  # (3, H, W)

def run():
    df = pd.DataFrame(columns=['Autoencoder', 'State Detector', 'Health Assessment', 'Hybrid', 'Classic', 'True'])

    spec, labels = load_data_for_pipeline()

    autoencoder = load_learner("./models/autoencoder.pkl")
    sd = load_learner("./models/sd-model.pkl")
    ha = load_learner("./models/ha-model.pkl")
    hybrid = load_learner("./models/hybrid.pkl")
    classic = load_learner("./models/classic_cnn.pkl")

    report = Report(df, autoencoder, sd, ha, hybrid, classic)
    report.run_report(spec, labels)

    df.to_csv(Path('./reports/report.csv'))

run()