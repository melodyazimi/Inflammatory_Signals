import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, remove_ectopic_beats, remove_outliers, get_nn_intervals

# #time domain meaure - to measure variability between successive heartbeats
def rmssd(hr_csv):
    #create RR interval array to pass into numpy math for RMSSD (root mean squared for successive differences)
    RR_array = np.array(hr_csv["rr interval (ms)"])

    #each element from the array has the np.diff function used on it - returns array of RR interval differences
    RR_diff = np.diff(RR_array)
    #each element in the array gets the sqaured function applied to it
    square = RR_diff ** 2
    #get the mean of diff values 
    RR_mean = np.mean(square)
    #once you get the mean of everything 
    RR_rmssd = np.sqrt(RR_mean)

    return(RR_rmssd, RR_array)




def complete_rr(rr_df):
    rr_df.columns = ["rr interval (ms)"]
    rr_df["rr interval (ms)"] = rr_df["rr interval (ms)"].astype(int)
    rr_intervals_list = rr_df["rr interval (ms)"].tolist()

    return(rr_intervals_list)




def pre_process_ecg(ecg_df):
    ecg_df.columns = ["microvolts"]

    #convert microvolts to milivolts
    ecg_df["milivolts"] = ecg_df["microvolts"] / 1000 

    #application takes information every 7.692 miliseconds
    ecg_df["time (ms)"] = ecg_df.index * 7.692

    #preprocess to remove noise - butterworth filter (type of signal processing filter)
    b, a = butter(4, fs = 1000)
    filtered_ecg = filtfilt(b, a, ecg_df)
    # print(filtered_ecg)




def find_peaks_ecg(ecg_df):
    peaks, properties = find_peaks(ecg_df["milivolts"], height = 0, distance=150)

    plt.figure(figsize=(10, 6))
    plt.plot(ecg_df["time (ms)"], ecg_df["milivolts"], label='ECG Signal')
    plt.plot(ecg_df["time (ms)"].iloc[peaks], ecg_df["milivolts"].iloc[peaks], 'x', color='red', label='Detected Peaks')
    plt.title('Detected R-peaks in ECG Signal')
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude (millivolts)')
    plt.legend()
    plt.show()



    first_5_seconds = ecg_df[(ecg_df['time (ms)'] >= 2000) & (ecg_df["time (ms)"] <=3000)]
    # Plotting the first 5000 milliseconds
    plt.figure(figsize=(10, 4))
    plt.plot(first_5_seconds['time (ms)'], first_5_seconds['milivolts'])  # Adjusted column names
    plt.title('First 5 Seconds of ECG')
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('Amplitude (mV)')
    plt.grid(True)
    plt.show()




def get_features(rr_interval_list):
    #remove all outliers, ectopic heart beats, and na values from the rr intervals
    nn_interval = get_nn_intervals(rr_interval_list)

    #get different features to understand HRV
    time_domain = get_time_domain_features(nn_interval)
    freq_domain = get_frequency_domain_features(nn_interval)

    return(time_domain, freq_domain)



##########################################################################################################
ecg_mel = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-17_13-31-20_ECG-data.txt", header = None)
rr_mel = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-17_13-31-22_RR-data.txt", header = None)

# ecg_mel = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-18_11-45-52_ECG-data.txt", header = None)
# rr_mel = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-18_11-45-53_RR-data.txt", header = None)

# ecg_allie = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-17_14-53-57_ECG-data.txt", header = None)
# rr_allie = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-17_14-53-58_RR-data.txt", header = None)

ecg_allie = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-18_11-36-48_ECG-data.txt", header = None)
rr_allie = pd.read_csv("/Users/melodyazimi/Downloads/VScode/2024-04-18_11-36-51_RR-data.txt", header = None)

# #ecg data
# ecg_data_allie = complete_ecg(ecg_allie)
# ecg_data_mel = complete_ecg(ecg_mel)
# # print(ecg_data_allie)
# print(ecg_data_mel)


#create rr interval data
rr_interval_mel = complete_rr(rr_mel)
rr_interval_allie = complete_rr(rr_allie)

time_domain_mel, freq_domain_mel = get_features(rr_interval_mel)
time_domain_allie, freq_domain_allie = get_features(rr_interval_allie)


columns = ["name", "Standard Deviation", "RMSSD", "PNN50", "HRV_HF"]
mel_data_point = {
    "name": ["melody"], 
    "Standard Deviation": [time_domain_mel["sdnn"]],
    "RMSSD": [time_domain_mel["rmssd"]], 
    "PNN50": [time_domain_mel["pnni_50"]], 
    "HRV_HF": [freq_domain_mel["hf"]]
}

allie_data_point = {
    "name": ["allie"],
    "Standard Deviation": [time_domain_allie["sdnn"]],
    "RMSSD": [time_domain_allie["rmssd"]],
    "PNN50": [time_domain_allie["pnni_50"]],
    "HRV_HF": [freq_domain_allie["hf"]]
}

mel_data = pd.DataFrame(mel_data_point, columns=columns)
mel_data.to_csv("mel_datapoint.csv", index = False)

allie_data = pd.DataFrame(allie_data_point, columns=columns)
allie_data.to_csv("allie_datapoint.csv", index = False)


# print(time_domain_mel)
# print("****")
# print(time_domain_allie)


# print(freq_domain_mel)
# print("*****")
# print(freq_domain_allie)












# rr_rmssd = rmssd(rr) #returns in ms
# print(rr_rmssd)

# ecg.columns = ["microvolts"]
# rr.columns = ["rr interval (ms)"]

# rr["rr interval (ms)"] = rr["rr interval (ms)"].astype(int)

# rr_intervals_list = rr["rr interval (ms)"].tolist()

# #convert microvolts to milivolts
# ecg["milivolts"] = ecg["microvolts"] / 1000

# #application takes information every 0.007692 miliseconds
# ecg["time (ms)"] = ecg.index * 7.692


# plt.figure(figsize = (20, 7))
# plt.plot(ecg["time (ms)"], ecg["milivolts"]) 
# plt.title("ECG over time")
# plt.xlabel("Time)")
# plt.ylabel("ECG")
# plt.show()


# first_5_seconds = ecg[(ecg['time (ms)'] >= 2000) & (ecg["time (ms)"] <=3000)]
# # Plotting the first 5000 milliseconds
# plt.figure(figsize=(10, 4))
# plt.plot(first_5_seconds['time (ms)'], first_5_seconds['milivolts'])  # Adjusted column names
# plt.title('First 5 Seconds of ECG')
# plt.xlabel('Time (milliseconds)')
# plt.ylabel('Amplitude (mV)')
# plt.grid(True)
# plt.show()





