
from sklearn.preprocessing import MinMaxScaler
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from gluonts.transform.sampler import InstanceSampler
from typing import Optional,  Iterable
from accelerate import Accelerator
from torch.optim import AdamW
import torch.optim.lr_scheduler as lr_scheduler
from evaluate import load
from gluonts.time_feature import get_seasonality 
from IPython.display import clear_output
import os 
import torch
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from gluonts.time_feature import (
    time_features_from_frequency_str,
    TimeFeature,
    get_lags_for_frequency,
)
from copy import deepcopy
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
from transformers import PretrainedConfig
import pandas as pd
import numpy as np
from helper import scaleDf, dataSplitSeparate, dataSplitSeparateTestVal, train, getParameterPairs, create_dataloader, create_val_dataloader, create_test_dataloader
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import get_lags_for_frequency
from gluonts.time_feature import time_features_from_frequency_str
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import random 
from scipy.stats import uniform, loguniform
from transformers import AutoformerConfig, AutoformerForPrediction
from transformers import InformerConfig, InformerForPrediction
import time

total_start = time.time()

df_train = pd.read_csv("electric_train.txt", sep=';')
df_test = pd.read_csv("electric_test.txt", sep=';')
df_val = pd.read_csv("electric_val.txt", sep=';')
print(len(df_test))
df_test = df_test.iloc[365:]
print(len(df_test))
df_train = df_train.replace(0.0, np.nan)
df_test = df_test.replace(0.0, np.nan)
df_val = df_train.replace(0.0, np.nan)

# Minmax scale each house independently
print("Min max scaling data...")
train_scaler_map = scaleDf(df_train)
test_scaler_map = scaleDf(df_test)
val_scaler_map = scaleDf(df_val)
# prediction_length = 7
prediction_length = 14  
# prediction_length = 17  
# prediction_length = 21  
# prediction_length = 28 # 4 weeks
# prediction_length = 42 # 6 weeks  
# prediction_length = 56 # 8 weeks  
freq = "1D" # set frequency to 1 day

print("Splitting data into train, val, test...")
train_dataset = dataSplitSeparate(df_train, freq, prediction_length)
# test_dataset, offset_map, ground_truth = dataSplitSeparateTestVal(df_test, freq, prediction_length)
test_dataset = dataSplitSeparate(df_test, freq, prediction_length)
val_dataset = dataSplitSeparate(df_val, freq, prediction_length)
# The look back window, in this case how many days we consider 
# lags_sequence = get_lags_for_frequency(freq)
lags_sequence = get_lags_for_frequency(freq)[:19]
time_features = time_features_from_frequency_str(freq)

config = TimeSeriesTransformerConfig(
    prediction_length=prediction_length,
    # # How many inputs the model take into account when 
    context_length=prediction_length * 2, 
    # lags coming from helper given the freq:
    lags_sequence=lags_sequence,# TODO: understand this 
    # day of year, day of months, and day of week
    num_time_features=len(time_features),
    # we have a single static categorical feature, namely time series ID:
    num_static_categorical_features=1,
    cardinality=[len(train_dataset)],
    # the model will learn an embedding of size 2 for each of the 366 possible values:
    embedding_dimension=[2],# TODO: understand this 
    scaling = "std",
    # transformer params:
    encoder_layers=4,
    decoder_layers=4,
    d_model=32, 
    num_parallel_samples = 1 # default
    
)

print("Creating dataloaders...")
train_dataloader = create_dataloader(
    "train",
    True,
    config=config,
    freq=freq,
    data=train_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

val_dataloader1 = create_dataloader(
    "validation",
    False,
    config=config,
    freq=freq,
    data=val_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader1 = create_dataloader(
    "test",
    False,
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=256,
    num_batches_per_epoch=100,
)

test_dataloader = create_test_dataloader(
    config=config,
    freq=freq,
    data=test_dataset,
    batch_size=64,
)



def getInformerConfig(encoder_layers, decoder_layers, d_model):
    
    config = InformerConfig(
        prediction_length=prediction_length,
        # # How many inputs the model take into account when 
        context_length=prediction_length * 2, 
        # lags coming from helper given the freq:
        lags_sequence=lags_sequence,# TODO: understand this 
        # day of year, day of months, and day of week
        num_time_features=len(time_features),
        # we have a single static categorical feature, namely time series ID:
        num_static_categorical_features=1,
        cardinality=[len(train_dataset)],
        # the model will learn an embedding of size 2 for each of the 366 possible values:
        embedding_dimension=[2],# TODO: understand this 
        scaling = "std",
        # transformer params:
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        d_model=d_model, 
        num_parallel_samples = 1 # default
        
    )
    
    return config

# BEST_EPOCH = 11 # for 7 
BEST_EPOCH = 28 # for 14 
# BEST_EPOCH = 45 # for 17 
# BEST_EPOCH = 36 # for 21
# BEST_EPOCH = 35 # for 28
# BEST_EPOCH = 43 # for 42
# BEST_EPOCH = 28 # FOR GAUSSIAN
BEST_LR = 0.005 
BEST_WD = 0.0001
BEST_EL = 16
BEST_DL = 4
BEST_DIMENSION = 32

informer_path = f"weights/electric/{str(prediction_length)}/separate/informer/{str(BEST_LR)}-{str(BEST_WD)}-{str(BEST_EL)}-{str(BEST_DL)}-{str(BEST_DIMENSION)}/state/{str(BEST_EPOCH)}"
informer_model = InformerForPrediction(getInformerConfig(BEST_EL, BEST_DL, BEST_DIMENSION))
print("CUDA IS AVAILABLE?:", torch.cuda.is_available())
informer_model.load_state_dict(torch.load(informer_path, map_location=torch.device('cpu')))
informer_model.eval()

def evaluate(model, data_loader):
    model.eval()

    forecasts = []
    for batch in data_loader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"]
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"]
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"],
            past_values=batch["past_values"],
            future_time_features=batch["future_time_features"],
            past_observed_mask=batch["past_observed_mask"],
        )
        forecasts.append(outputs.sequences.cpu().numpy())
    forecasts = np.vstack(forecasts)
    return forecasts

informer_forecasts = evaluate(informer_model, test_dataloader)

# Inverse scale the df back to its original values 
df_scaled_no_nan = deepcopy(df_test)
i = 0
for column_name in df_scaled_no_nan.columns:
    if column_name[0] != "M":
        continue
    curSeries = df_scaled_no_nan[column_name].values.reshape(-1, 1)
    df_scaled_no_nan[column_name] = test_scaler_map[i].inverse_transform(curSeries)
    i += 1
df_scaled_no_nan = df_scaled_no_nan.fillna(0)
NUM_OF_HOUSES = 106 
# OFFSET = 365
OFFSET = 365
# We also need to inverse scale the forecasts
# def scale_independently(forecasts, ctx_length):

#     # Offset map is a list of lists, where offset_map[i] tells your 
#     # how many entries of house i is not empty 
#     j = 0
#     for k in range(NUM_OF_HOUSES):
        
#         curScaler = test_scaler_map[k] 
#         for i in range(0, OFFSET, OFFSET - ctx_length):
#             forecasts[i + j] = curScaler.inverse_transform(forecasts[i + j].reshape(-1, 1)).reshape(-1)
#             j += 1
def scale_independently(forecasts, offset):
    
    for i in range(0, len(forecasts), offset):
        curScaler = test_scaler_map[i // offset] 
        for j in range(offset):
            # print(curScaler.inverse_transform(forecasts[i + j].reshape(-1, 1)).reshape(-1))
            forecasts[i + j] = curScaler.inverse_transform(forecasts[i + j].reshape(-1, 1)).reshape(-1)
            # curList.append(np.array(vanilla_forecasts[i + j][0]))
            
            
informer_forecasts_ev = deepcopy(informer_forecasts)
scale_independently(informer_forecasts_ev, len(df_test) - prediction_length * 4)

informer_forecasts_ev[np.isnan(informer_forecasts_ev)] = 0

# Transform test dataset's nan values to 0
for item_id, ts in enumerate(test_dataset): 
    for i in range(len(ts["target"])):
        if np.isnan(ts["target"][i]):
            test_dataset[item_id]["target"][i] = 0

from helper import kl_between_two_dist

# Gather all the forecasts and put in B
# Also gather all the forecasts' respective true value in A 

# TODO: Might have a bug here
print("Calcualting multivariate KL divergence between forecasted values and observed values...")
# def getForecastsAndObserved(forecasts):
#     A = []
#     B = []  
#     k = 0 
#     offset = 0
#     offset_gt = 0
#     for k in range(NUM_OF_HOUSES):
#         numOfTimepoints = len(forecasts) - OFFSET
#         curListA = []
#         curListB = [] 
#         for i in range(numOfTimepoints - prediction_length * 5):
#             curListA.append(np.array(ground_truth[offset_gt + i]))       
#             curListB.append(np.array(forecasts[offset + i][0]))

#             # 14
#         # 106 
#         # 125 - 28 
        
#         offset_gt += offset_map[k] - prediction_length * 5 
#         offset += offset_map[k] - prediction_length * 4 
#         A.append(np.array(curListA))   
#         print(A[-1].shape)
#         B.append(np.array(curListB))        
#         print(B[-1].shape)

#     return A, B

def getForecastsAndObserved(forecasts):
    A = []
    B = []
    for column_name in df_scaled_no_nan.columns:
        curList = [] 
        if column_name[0] != "M":
            continue
        cur_df = df_scaled_no_nan[column_name][prediction_length * 4:]
        for i in range(len(cur_df) - prediction_length):
            curList.append(np.array(cur_df[i : i + prediction_length]))        
        A.append(np.array(curList))
        
    for i in range(NUM_OF_HOUSES):
        curList = [] 
        for j in range(len(cur_df) - prediction_length):
            curList.append(np.array(forecasts[i * len(cur_df) + j][0]))
            
        B.append(np.array(curList))
    
    A = np.array(A)
    B = np.array(B)
    return A, B
# Calculate the KL divergences between these two Multivariate gaussian distribution 
# for every single forecasts (97 forecasts)

# A = [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2]]

def plotKL_Div_between_A_and_B(A, B, save_name, title):
    KL_div_list = [] 

    # Calcualte the KL div between all 97 forecast points and observed points
    for i in range(len(A)):
        KL_div_list.append(kl_between_two_dist(A[:, i, :], B[:, i, :]))
        
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.plot(KL_div_list)
    plt.savefig(save_name)

    return KL_div_list

informerA, informerB = getForecastsAndObserved(informer_forecasts_ev)

df_pair = pd.DataFrame({'true': informerA.flatten(), 'pred': informerB.flatten()})

# Step 4: Save the DataFrame to a file (e.g., CSV file)
df_pair.to_csv('pair.csv', index=False)
print("flatten complete.")

informer_KL_div_list = plotKL_Div_between_A_and_B(informerA, informerB, "kl_divergence_separate.png", "KL divergence for each forecasts (Informer)")
from helper import l1_distances, l1_distances_mean

def get_l1_distances_list(A, B):
    l1_distances_list = []  # 970

    # Calcualte the KL div between all 97 forecast points and observed points
    for i in range(A.shape[1]):
        l1_distances_list.extend(l1_distances(A[:, i, :], B[:, i, :]))
        
    plt.figure(figsize=(16, 8))
    plt.title("l1 distances")
    plt.plot(l1_distances_list)
    plt.savefig("l1_distances_separate.png")
    return l1_distances_list

def get_l1_distances_mean_list(A, B):
    l1_distances_mean_list = []  # 970
    for i in range(A.shape[1]):
        l1_distances_mean_list.append(l1_distances_mean(A[:, i, :], B[:, i, :]))
    plt.figure(figsize=(16, 8))
    plt.title("l1 distances mean per " + str(NUM_OF_HOUSES) + " houses")
    plt.plot(l1_distances_mean_list)
    plt.savefig("l1_distances_mean_separate.png")
    return l1_distances_mean_list
l1dlinf = get_l1_distances_list(informerA, informerB)
l1dlinfm = get_l1_distances_mean_list(informerA, informerB)

print("scale_dataset...")
def scale_dataset(dataset):
    offset = len(dataset) // NUM_OF_HOUSES
    for i in range(0, len(dataset), offset):
        curScaler = test_scaler_map[i // offset] 
        for j in range(offset):
            dataset[i + j]["target"] = np.array(curScaler.inverse_transform(dataset[i + j]["target"].reshape(-1, 1)).reshape(-1))
            
test_dataset_ev = deepcopy(test_dataset)
scale_dataset(test_dataset_ev)
for i in range(len(test_dataset_ev)):
    test_dataset_ev[i]["target"][np.isnan(test_dataset_ev[i]["target"])] = 0
from helper import CRPS, crps_nrg, crps_pwm
from gluonts.time_feature import get_seasonality
def calculate(forecasts, dataset):

    mse_metric = load("evaluate-metric/mse")
    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1)

    mse_metrics = []
    mase_metrics = []
    smape_metrics = []
    crps_metrics = []
    for item_id, ts in enumerate(dataset):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        mse = mse_metric.compute(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth)
            )
        mse_metrics.append(mse["mse"])
        
        mase = mase_metric.compute(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
            training=np.array(training_data), 
            periodicity=get_seasonality(freq))
        mase_metrics.append(mase["mase"])
        
        smape = smape_metric.compute(
            predictions=forecast_median[item_id], 
            references=np.array(ground_truth), 
        )
        smape_metrics.append(smape["smape"])

        crps_metrics.append(CRPS(forecast_median[item_id], np.array(ground_truth)))
        
    return mse_metrics, mase_metrics, smape_metrics, crps_metrics


print("Getting scores...")
mse_metrics4, mase_metrics4, smape_metrics4, crps_metrics4  = calculate(informer_forecasts_ev, test_dataset_ev)  

with open("informer_scores_separate.txt", "a") as f:
     
    f.write(f"Informer Test MSE: {np.mean(mse_metrics4)}\n")
    f.write("===============================================\n")
    f.write(f"Informer Test MASE: {np.mean(mase_metrics4)}\n")
    f.write("===============================================\n")
    f.write(f"Informer Test sMAPE: {np.mean(smape_metrics4)}\n")
    f.write("===============================================\n")
    f.write(f"Informer Test crps: {np.mean(crps_metrics4)}\n")  
import matplotlib.dates as mdates
import random 
test_dataset_ev = np.array(test_dataset_ev)
def plot(ts_index, forecasts, save = False):
    fig, ax = plt.subplots()

    index = pd.period_range(
        start=test_dataset_ev[ts_index][FieldName.START],
        periods=len(test_dataset_ev[ts_index][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()
    index = index.to_numpy()
    # Major ticks every half year, minor ticks every month,
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.plot(
        index, 
        test_dataset_ev[ts_index]["target"],
        label="actual",
    )

    plt.title(str(ts_index) + "-" + str(int(ts_index) + prediction_length))
    plt.plot(
        index[-prediction_length:], 
        np.median(forecasts[ts_index], axis=0),
        label="median",
    )
    
    plt.fill_between(
        index[-prediction_length:],
        forecasts[ts_index].mean(0) - forecasts[ts_index].std(axis=0), 
        forecasts[ts_index].mean(0) + forecasts[ts_index].std(axis=0), 
        alpha=0.3,  
        interpolate=True,
        label="+/- 1-std",
    )
    if save:
        if not os.path.exists(f"graph/electric/{str(prediction_length)}/separate/informer2/"):
            os.makedirs(f"graph/electric/{str(prediction_length)}/separate/informer2/")
        plt.savefig(f"graph/electric/{str(prediction_length)}/separate/informer2/" + str(ts_index) + "-" + str(int(ts_index) + prediction_length) + ".png")
    plt.legend()
    plt.show()

def plot_randomly(forecasts):
    plot(random.randint(0, len(forecasts) - 1), forecasts, True)

print("Plotting graphs...")
for _ in range(100):
    plot_randomly(informer_forecasts_ev)
    
total_end = time.time()
print("The whole program took: " + str(total_end - total_start) + "s")