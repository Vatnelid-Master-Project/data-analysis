from fastai.vision.all import *

from make import load_test_data_for_pipeline
from collections import Counter

FLOOR_DB = -45.0
TARGET_HW = (112, 112)

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
    
    def run_vote(self, ha_vote, hybrid_vote, classic_vote):
        result_list = []

        result_list.append(ha_vote)
        result_list.append(hybrid_vote)
        result_list.append(classic_vote)

        vote_winner = Counter(result_list)

        if (vote_winner.most_common(1)[0][1] == 1):
            return ha_vote

        return vote_winner.most_common(1)[0][0]




    def run_report(self, specs: list, true_labels: list):
        df = self.dataframe
        
        # Run autoencoder
        for i in range(len(specs)):
            auto_result = self.run_autonencoder(self.__get_tensor(specs[i]), 0.013841685838997364)
            sd_result = self.run_state_detection(self.__get_tensor(specs[i]))
            
            if auto_result == "Fall" and sd_result == "Fall":
                ha_result = self.run_heath_assessment(self.__get_tensor(specs[i]))
                hybrid_result = self.run_hybrid(self.__get_tensor(specs[i]))
                classic_result = self.run_classic(self.__get_tensor(specs[i]))
                vote_winner = self.run_vote(ha_result, hybrid_result, classic_result)
            else:
                ha_result = "Not Fall"
                hybrid_result = "Not Fall"
                classic_result = "Not Fall"
                vote_winner = "Not Fall"

            df.loc[len(df)] = [auto_result, sd_result, ha_result, hybrid_result, classic_result, vote_winner, true_labels[i]]
    
    def __get_tensor(self, arr):
        # If spectrogram is multi-channel, convert to grayscale first
        if arr.ndim == 3:
            arr = np.mean(arr, axis=-1)  # (H, W)

        # Ensure float32
        arr = arr.astype(np.float32)

        # If not already in dB, convert. If it *is* already in dB, skip this block.
        #arr = np.maximum(arr, EPS)
        #arr_db = 10.0 * np.log10(arr)

        # Apply dynamic range floor and normalize to [0,1]
        arr = arr - arr.max()
        arr = np.clip(arr, FLOOR_DB, 0.0)
        arr_01 = (arr - FLOOR_DB) / (-FLOOR_DB)

        arr_01 = arr_01.astype(np.float32)

        arr_01 = arr_01 ** 0.9

        arr_01 = np.flipud(arr_01)

        cmap = plt.get_cmap('magma')
        img_rgb = cmap(arr_01)[:, :, :3]


        # Convert to tensor and make 3-channel (grayscale -> RGB-style)
        t = torch.from_numpy(img_rgb)          # (H, W)
        t = t.float().permute(2, 0, 1).unsqueeze(0)    # (3, H, W)

        # Add batch dimension for interpolate
        #t = t.unsqueeze(0)                    # (1, 3, H, W)

        # Resize
        t_resized = F.interpolate(
                t, size=TARGET_HW, mode='bilinear', align_corners=False
        )

        return TensorImage(t_resized.squeeze(0))  # (3, H, W)

def run():
    df = pd.DataFrame(columns=['Autoencoder', 'State Detector (Resnet18)', 'Health Assessment (ResNet34)', 'Hybrid', 'Classic', 'Vote', 'True'])

    spec, labels = load_test_data_for_pipeline()

    autoencoder = load_learner("./models/autoencoder.pkl")
    sd = load_learner("./models/sd-model.pkl")
    ha = load_learner("./models/ha-model.pkl")
    hybrid = load_learner("./models/hybrid.pkl")
    classic = load_learner("./models/classic_cnn.pkl")

    report = Report(df, autoencoder, sd, ha, hybrid, classic)
    report.run_report(spec, labels)

    df.to_csv(Path('./reports/report.csv'))

run()