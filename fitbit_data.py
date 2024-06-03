##environment = ml_env

import pandas as pd
from hrvanalysis import get_time_domain_features, get_frequency_domain_features, remove_ectopic_beats, remove_outliers, get_nn_intervals
from datetime import datetime, timedelta, time
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm, metrics, preprocessing
from sklearn.tree import DecisionTreeClassifier
# import tensorflow as tf

def get_rest(sleep_data):
    columns = ['Id', 'sleep_start', 'sleep_end']
    sleep_times = pd.DataFrame(columns=columns) #create the empty dataframe

    sleep_times_list = []

    sleep_dict = {}
    for person_id, group in sleep_data.groupby(["Id", "logId"]):
        #find start time for sleep
        sleep_start = group["date"].iloc[0]
        sleep_end = group["date"].iloc[-1]

        #only add the first instance of sleep
        if person_id[0] not in sleep_dict:

        # next_time_period = sleep_start + timedelta(minutes=1) #look at the consective time slot
        # # print(sleep_start, next_time_period)

        # for time_slot in group["date"]:
        #     #check if the next time slot is consecutive to the previous time slot 
        #     if time_slot == next_time_period:
        #         next_time_period = time_slot + timedelta(minutes=1) #if it is, keeping adding another minutee
        #     else:
        #         #if the next one isn't conecutive, then remove the addition of the minute to get the end of their sleep cycle
        #         sleep_end = next_time_period - timedelta(minutes=1) 

            # sleep_dict[person_id[0]] = sleep_start, sleep_end

            #stores each row in the dictionary
            sleep_times_list.append({'Id': person_id[0], 'sleep_start': sleep_start, 'sleep_end': sleep_end})


            # sleep_times = sleep_times[{new_data}] #the new_data is a dictionary that needs to be made into a dataframe 
    
    #makes the list into a dataframe
    sleep_times = pd.DataFrame(sleep_times_list, columns=columns)
    return(sleep_times)


def get_features(rr_interval_list):
    #remove all outliers, ectopic heart beats, and na values from the rr intervals
    nn_interval = get_nn_intervals(rr_interval_list)

    #get different features to understand HRV
    time_domain = get_time_domain_features(nn_interval)
    freq_domain = get_frequency_domain_features(nn_interval)

    return(time_domain, freq_domain)



def find_features(heart_df):
    new_data_point = [] #empty list to append to later 

    for id, group in heart_df.groupby("Id"):
        group["rr_interval"] = 60000/(group["Value"]) #calculate rr interval from bpm
        rr_interval_list = group["rr_interval"].tolist() #create list of rr intervals for "get features" function

        #calculate features: time domain and frequency domain features
        time_domain, freq_domain = get_features(rr_interval_list) #call function from Capstone.py 

        #add each new datapoint to a list with the id and correpsonding hrv features
        new_data_point.append({"id": id, "rmssd": time_domain["rmssd"], "SD": time_domain["sdnn"], 
                            "PNN50": time_domain["pnni_50"], "HF": freq_domain["hf"]})
        

    #update a new dataframe with the feature information
    columns = ["id", "rmssd", "SD", "HF"]
    fitbit_data = pd.DataFrame(new_data_point, columns = columns)

    return(fitbit_data)




def svm_test (X_train, X_test, Y_train, Y_test, X, Y):
    svc_model = svm.SVC(kernel="linear")
    svc_model.fit(X_train, Y_train)

    y_preds_svc = svc_model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, y_preds_svc)
    classif = metrics.classification_report(Y_test, y_preds_svc)
    print("svm accuracy:", accuracy)
    print(classif)

    cross_validation(svc_model, X, Y)




def knn_test(X_train, X_test, Y_train, Y_test, X, Y):
    knn_model = KNeighborsClassifier(n_neighbors=3) #checks to see what its closest value is to its 3 nearest neighbors
    knn_model.fit(X_train, Y_train)

    y_preds_knn = knn_model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, y_preds_knn)
    classif = metrics.classification_report(Y_test, y_preds_knn)
    print("Knn accuracy:", accuracy)
    print(classif)

    cross_validation(knn_model, X, Y)
    return(knn_model)



def decisionTree_test(X_train, X_test, Y_train, Y_test, X, Y):
    dt_model = DecisionTreeClassifier(random_state=123)
    dt_model.fit(X_train, Y_train)

    y_preds_dt = dt_model.predict(X_test)

    accuracy = metrics.accuracy_score(Y_test, y_preds_dt)
    print("Decision Tree accuracy:", accuracy)
    conf_matrix = metrics.confusion_matrix(Y_test, y_preds_dt)
    class_report = metrics.classification_report(Y_test, y_preds_dt)
    print(conf_matrix)
    print(class_report)

    cross_validation(dt_model, X, Y)
    return(dt_model)




def cross_validation(model, X, y):
    #cross-validation - to make sure the model is overfitted and assess the model's performance
    cv = cross_val_score(model, X, y, cv=5) #5 fold cross validation
    print("Accuracy average scores for each fold:", cv.mean())

    #stratified is helpful to get the same proportion of each label in the group - helpful for uneven datasets
    strat = StratifiedKFold(n_splits = 5, shuffle=True, random_state=123)
    cv_strat = cross_val_score(model, X, y, cv = strat)
    print("Accuracy average scores for each strat fold:", cv_strat.mean())






###################################################################
#data from fitbit kaggle dataset
sleep_df = pd.read_csv("/Users/melodyazimi/Downloads/VScode/minuteSleep_merged.csv")
heart_df = pd.read_csv("/Users/melodyazimi/Downloads/VScode/heartrate_seconds_merged.csv")

#data from diabetes kaggle dataset
diabetes_df = pd.read_csv("/Users/melodyazimi/Downloads/VScode/new_dataset_data_research_new_pred.csv")

#process data - sleep
sleep_df["date"] = pd.to_datetime(sleep_df["date"], format = "%m/%d/%Y %I:%M:%S %p")
sleep_df["Id"] = sleep_df["Id"].astype(int)

#process data = heart
heart_df["Time"] = pd.to_datetime(heart_df["Time"], format = "%m/%d/%Y %I:%M:%S %p")
heart_df["Id"] = heart_df["Id"].astype(int)
heart_df["Value"] = heart_df["Value"].astype(int)

# fitbit_df = find_features(heart_df)

#Data cleaning and feature selection
diabetes_df = diabetes_df[["Standard Deviation", "RMSSD", "PNN50", "HRV_HF", "Diabetic"]]
narows = diabetes_df.isna().any(axis=1)

#split data into labeled and features
X = diabetes_df.drop("Diabetic", axis=1) #axis = 1 means it is a column, and to drop "Diabetic"
Y = diabetes_df["Diabetic"]

#this shows that there is not an even amount of diabettic to non-diabetic - so we need to scale the data to make it fair
print(len(diabetes_df[diabetes_df["Diabetic"] == 0.0]))
print(len(diabetes_df[diabetes_df["Diabetic"] == 1.0]))

#split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.8, random_state=123)
print(X_train)

#scale the data for normalization - best for classification
scale = preprocessing.StandardScaler()
X_train_scaled = scale.fit_transform(X_train) #best to fit the scale on training data 
X_test_scaled = scale.transform(X_test)

#min max scaler used for neural networks - helps with activation function since those functions are used on a 0 to 1 scale so normalization is important
scale_nn = preprocessing.MinMaxScaler()
X_train_scaled_nn = scale.fit_transform(X_train)
X_test_scaled_nn = scale.transform(X_test)

scale_rr = preprocessing.RobustScaler()
X_train_scaled_rr = scale_rr.fit_transform(X_train)
x_test_scaled_rr = scale_rr.transform(X_test)


print(diabetes_df)
svm_test(X_train, X_test, Y_train, Y_test, X, Y) #og
# svm_test(X_train_scaled, X_test_scaled, Y_train, Y_test, X, Y) #standard

knn_model = knn_test(X_train, X_test, Y_train, Y_test, X, Y)#og
# knn_test(X_train_scaled, X_test_scaled, Y_train, Y_test, X, Y) #standard
# knn_test(X_train_scaled_nn, X_test_scaled_nn, Y_train, Y_test, X, Y) #minmax
knn_test(X_train_scaled_rr, x_test_scaled_rr, Y_train, Y_test, X, Y) #robust

dectree_model = decisionTree_test(X_train, X_test, Y_train, Y_test, X, Y)



######
mel_data_point = pd.read_csv("mel_datapoint.csv")
allie_data_point = pd.read_csv("allie_datapoint.csv")

mel_data_point = mel_data_point.drop("name", axis = 1)
allie_data_point = allie_data_point.drop("name", axis = 1)

mel_prediction = dectree_model.predict(mel_data_point)
# mel_prediction = knn_model.predict(mel_data_point)
allie_prediction = dectree_model.predict(allie_data_point)
print(f"The predicted class for the new data point is: {mel_prediction}")
print(f"The predicted class for the new data point is: {allie_prediction}")




# #tensorflow
# tf_model = tf.keras.Sequential([
#     tf.keras.layers.Dense(16, activation = "relu"), #dense neurnets - a layer of 16 neurons with the activation function called relu
#     tf.keras.layers.Dense(16, activation = "tanh"),
#     tf.keras.layers.Dense(1, activation = "sigmoid")
#  ])


#cross validation


# fitbit_df = find_features(heart_df)





# print(heart_df)

    # time_domain, freq_domain = get_features(rr_interval_list)
    # print(id)
    # print(time_domain)






    # rr_interval_array = np.diff(group["Value"])
    # print(rr_interval_array)






# print(heart_df)
# sleep_times = get_rest(sleep_df)
# print(sleep_times)

# #create a set of ids - set makes sure there are no duplicates and makes it easy to check if something is in a set whereas with a list you need to iterate over them
# sleep_ids = set(sleep_times["Id"])
# heart_ids = set(heart_df["Id"])

# #find the ids that are common in both dataframes
# common_ids = sleep_ids.intersection(heart_ids)

# filtered_heart_df = heart_df[heart_df["Id"].isin(common_ids)] #only keep the ids that are common


    











#     # sleep_end = datetime.strptime(group["date"].iloc[-1], "%m/%d/%Y %I:%M:%S %p")
#     # sleep_end = group["date"].iloc[-1]
#     print(person_id, sleep_start)

#     for time in group["date"]:

#         next_time_period = sleep_start + timedelta(minutes=1)
#         if sleep_start - 
    

    # next_day = sleep_start.date() + timedelta(days = 1)
    # before_noon = next_day + timedelta(hours = 12)
    # for time in group["date"]:
    #     if time.date()
    #     print(before_noon)
    #     # if time.date() == next_day and time.time() < 12:00:00:
    #     #     next_day_list.append(time)
    #     #     print(next_day_list[-1])


