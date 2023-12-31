
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
from helper import scaleDf, dataSplit, train, linf_distances, getParameterPairs, create_dataloader, create_val_dataloader, create_test_dataloader
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

# df_days = pd.read_csv("electric_day.txt", sep=';')
df_days = pd.read_csv("electricity_day_clean.csv")
print("Length of dataset", len(df_days))
# Replace all 0.0s with NaN, this is needed for the observed mask that will be passed into the transformer
df_days = df_days.replace(0.0, np.nan)
df_scaled = df_days

# Minmax scale each house independently
print("Min max scaling data...")
scaler_map = scaleDf(df_scaled)
num_of_datapoints = len(df_scaled)
train_size = int(num_of_datapoints * 0.5)
val_size = int(0.75 * num_of_datapoints) - train_size
prediction_length = 14 # 7 days 
freq = "1D" # set frequency to 1 day

print("Splitting data into train, val, test...")
train_dataset, val_dataset, test_dataset = dataSplit(df_scaled, freq, prediction_length)

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

BEST_EPOCH = 15
BEST_LR = 0.0013974721025626093
BEST_WD = 0.005139748702619967
BEST_EL = 32
BEST_DL = 16
BEST_DIMENSION = 64

informer_path = f"weights/electric_clean/14/informer/{BEST_LR}-{BEST_WD}-{BEST_EL}-{BEST_DL}-{BEST_DIMENSION}/state/{str(BEST_EPOCH)}"
informer_model = InformerForPrediction(getInformerConfig(BEST_EL, BEST_DL, BEST_DIMENSION))
informer_model.load_state_dict(torch.load(informer_path))
# informer_model = InformerForPrediction()
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
print("SHAPE IS:", informer_forecasts.shape)
# Inverse scale the df back to its original values 
df_scaled_no_nan = deepcopy(df_scaled)
i = 0
for column_name in df_scaled_no_nan.columns:
    if column_name[0] != "M":
        continue
    curSeries = df_scaled_no_nan[column_name].values.reshape(-1, 1)
    df_scaled_no_nan[column_name] = scaler_map[i].inverse_transform(curSeries)
    i += 1
df_scaled_no_nan = df_scaled_no_nan.fillna(0)

# We also need to inverse scale the forecasts
def scale_independently(forecasts, offset):
    
    for i in range(0, len(forecasts), offset):
        curScaler = scaler_map[i // offset] 
        for j in range(offset):
            # print(curScaler.inverse_transform(forecasts[i + j].reshape(-1, 1)).reshape(-1))
            forecasts[i + j] = curScaler.inverse_transform(forecasts[i + j].reshape(-1, 1)).reshape(-1)
            # curList.append(np.array(vanilla_forecasts[i + j][0]))
        
informer_forecasts_ev = deepcopy(informer_forecasts)
test_size = len(df_days) - (train_size +  val_size)
offset = test_size - prediction_length * 4
scale_independently(informer_forecasts_ev, offset)
informer_forecasts_ev[np.isnan(informer_forecasts_ev)] = 0
# (261, 14, 320)
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
NUM_OF_HOUSES = 320

USE_TSMIXER = False
def getForecastsAndObserved(forecasts):
    A = []
    B = []
    for column_name in df_scaled_no_nan.columns:
        curList = [] 
        if column_name[0] != "M":
            continue
        cur_df = df_scaled_no_nan[column_name][train_size + val_size + prediction_length * 4:]
        for i in range(len(cur_df) - prediction_length):
            curList.append(np.array(cur_df[i : i + prediction_length]))        
        A.append(np.array(curList))
    
    if not USE_TSMIXER:
        for i in range(NUM_OF_HOUSES):
            curList = [] 
            for j in range(len(cur_df) - prediction_length):
                curList.append(np.array(forecasts[i * len(cur_df) + j][0]))
                
            B.append(np.array(curList))
    # else:

    #     import tensorflow as tf
    #     from tensorflow.keras import layers


    #     def res_block(inputs, norm_type, activation, dropout, ff_dim):
    #         """Residual block of TSMixer."""

    #         norm = (
    #             layers.LayerNormalization
    #             if norm_type == 'L'
    #             else layers.BatchNormalization
    #         )

    #         # Temporal Linear
    #         x = norm(axis=[-2, -1])(inputs)
    #         x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    #         x = layers.Dense(x.shape[-1], activation=activation)(x)
    #         x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    #         x = layers.Dropout(dropout)(x)
    #         res = x + inputs

    #         # Feature Linear
    #         x = norm(axis=[-2, -1])(res)
    #         x = layers.Dense(ff_dim, activation=activation)(
    #             x
    #         )  # [Batch, Input Length, FF_Dim]
    #         x = layers.Dropout(dropout)(x)
    #         x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    #         x = layers.Dropout(dropout)(x)
    #         return x + res


    #     def build_model(
    #         input_shape,
    #         pred_len,
    #         norm_type,
    #         activation,
    #         n_block,
    #         dropout,
    #         ff_dim,
    #         target_slice,
    #     ):
    #         """Build TSMixer model."""

    #         inputs = tf.keras.Input(shape=input_shape)
    #         x = inputs  # [Batch, Input Length, Channel]
    #         for _ in range(n_block):
    #             x = res_block(x, norm_type, activation, dropout, ff_dim)

    #         if target_slice:
    #             x = x[:, :, target_slice]

    #         x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    #         x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    #         outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

    #         return tf.keras.Model(inputs, outputs)
    #     model = build_model(
    #         input_shape=(prediction_length * 4, NUM_OF_HOUSES),
    #         pred_len=prediction_length,
    #         norm_type="B",
    #         activation="relu",
    #         dropout=0.7,
    #         n_block=4,
    #         ff_dim=64,
    #         target_slice= slice(0, None),
    #     )
    #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #     model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    #     model.load_weights("bernard_M_tsmixer_sl56_pl14_lr0.0001_ntB_relu_nb4_dp0.7_fd64_best.index")
    #     # test_result = model.evaluate(test_data)
    #     B = model.predict(test_data)
    #     B = np.transpose(B, (2, 0, 1))

    A = np.array(A)
    B = np.array(B)
    print("SHAPE IS 2:", A.shape, B.shape)
    return A, B
# Calculate the KL divergences between these two Multivariate gaussian distribution 
# for every single forecasts (97 forecasts)
   

# (261, 14, 320) -> (320, 261, 14)
def plotKL_Div_between_A_and_B(A, B, save_name, title):
    KL_div_list = [] 

    # Calcualte the KL div between all 97 forecast points and observed points
    for i in range(A.shape[1]):
        KL_div_list.append(kl_between_two_dist(A[:, i, :], B[:, i, :]))
        
    plt.figure(figsize=(5, 5))
    plt.title(title)
    plt.plot(KL_div_list)
    plt.savefig(save_name)

    return KL_div_list

informerA, informerB = getForecastsAndObserved(informer_forecasts_ev)
informer_KL_div_list = plotKL_Div_between_A_and_B(informerA, informerB, "kl_divergence_clean.png", "KL divergence (separating mean)")
from helper import l1_distances, l1_distances_mean

def get_l1_distances_list(A, B):
    l1_distances_list = []  # 970

    # Calcualte the KL div between all 97 forecast points and observed points
    for i in range(A.shape[1]):
        l1_distances_list.extend(l1_distances(A[:, i, :], B[:, i, :]))
        
    plt.figure(figsize=(16, 8))
    plt.title("l1 distances")
    plt.plot(l1_distances_list)
    plt.savefig("l1_distances_clean.png")
    return l1_distances_list

def get_l1_distances_mean_list(A, B):
    l1_distances_mean_list = []  # 970
    for i in range(A.shape[1]):
        l1_distances_mean_list.append(l1_distances_mean(A[:, i, :], B[:, i, :]))
    plt.figure(figsize=(16, 8))
    plt.title("l1 distances mean per 10 houses")
    plt.plot(l1_distances_mean_list)
    plt.savefig("l1_distances_mean_clean.png")
    return l1_distances_mean_list


def get_linf_distances_list(A, B):
    linf_distances_list = []  # 970

    # Calcualte the KL div between all 97 forecast points and observed points
    for i in range(A.shape[1]):
        linf_distances_list.extend(linf_distances(A[:, i, :], B[:, i, :]))
        
    plt.figure(figsize=(16, 8))
    plt.title("linf distances")
    plt.plot(linf_distances_list)
    plt.savefig("linf_distances_clean.png")
    return linf_distances_list

l1dlinf = get_l1_distances_list(informerA, informerB)
l1dlinfm = get_l1_distances_mean_list(informerA, informerB)
l1infd_inf = get_linf_distances_list(informerA, informerB)

print("scale_dataset...")
def scale_dataset(dataset):
    offset = len(dataset) // 10
    print(offset)
    for i in range(0, len(dataset), offset):
        curScaler = scaler_map[i // offset] 
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


print("Getting scores (after preprocessing)")
mse_metrics4, mase_metrics4, smape_metrics4, crps_metrics4  = calculate(informer_forecasts, test_dataset)  

with open("informer_scores_clean.txt", "a") as f:
     
    f.write(f"Informer Test MSE: {np.mean(mse_metrics4)}")
    f.write("===============================================")
    f.write(f"Informer Test MASE: {np.mean(mase_metrics4)}")
    f.write("===============================================")
    f.write(f"Informer Test sMAPE: {np.mean(smape_metrics4)}")
    f.write("===============================================")
    f.write(f"Informer Test crps: {np.mean(crps_metrics4)}")  
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
        if not os.path.exists("graph/electric_clean/14/informer/"):
            os.makedirs("graph/electric_clean/14/informer/")
        plt.savefig("graph/electric_clean/14/informer/" + str(ts_index) + "-" + str(int(ts_index) + prediction_length) + "_clean.png")
    plt.legend()
    plt.show()

def plot_randomly(forecasts):
    plot(random.randint(0, len(forecasts) - 1), forecasts, True)

print("Plotting graphs...")
for _ in range(100):
    plot_randomly(informer_forecasts_ev)
    
total_end = time.time()
print("The whole program took: " + str(total_end - total_start) + "s")