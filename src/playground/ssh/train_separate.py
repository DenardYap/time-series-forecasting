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

"""
TODO:
"""
df_train = pd.read_csv("electric_train.txt", sep=';')
df_test = pd.read_csv("electric_test.txt", sep=';')
df_val = pd.read_csv("electric_val.txt", sep=';')
df_train = df_train.replace(0.0, np.nan)
df_test = df_test.replace(0.0, np.nan)
df_val = df_train.replace(0.0, np.nan)

# Minmax scale each house independently
print("Min max scaling data...")
train_scaler_map = scaleDf(df_train)
test_scaler_map = scaleDf(df_test)
val_scaler_map = scaleDf(df_val)
# prediction_length = 14  
prediction_length = 28 # 4 weeks
# prediction_length = 42 # 6 weeks  
prediction_length = 56 # 8 weeks  
freq = "1D" # set frequency to 1 day

print("Splitting data into train, val, test...")
train_dataset = dataSplitSeparate(df_train, freq, prediction_length)
# test_dataset, ground_truth = dataSplitSeparateTestVal(df_test, freq, prediction_length)
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

# test_dataloader1 = create_dataloader(
#     "test",
#     False,
#     config=config,
#     freq=freq,
#     data=test_dataset,
#     batch_size=256,
#     num_batches_per_epoch=100,
# )

# test_dataloader = create_test_dataloader(
#     config=config,
#     freq=freq,
#     data=test_dataset,
#     batch_size=64,
# )


# batch = next(iter(train_dataloader))

# test_run_model = TimeSeriesTransformerForPrediction(config)
# test_run_model.eval()
# outputs = test_run_model(
#     past_values=batch["past_values"],
#     past_time_features=batch["past_time_features"],
#     past_observed_mask=batch["past_observed_mask"],
#     static_categorical_features=batch["static_categorical_features"]
#     if config.num_static_categorical_features > 0
#     else None,
#     static_real_features=batch["static_real_features"]
#     if config.num_static_real_features > 0
#     else None,
#     future_values=batch["future_values"],
#     future_time_features=batch["future_time_features"],
#     future_observed_mask=batch["future_observed_mask"],
#     output_hidden_states=True,
# )
# print("Test run loss:", outputs.loss.item())
# print("TEST RUN OK!")


def getConfig(encoder_layers, decoder_layers, d_model):
    
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
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        d_model=d_model, 
        num_parallel_samples = 1 # default
        
    )
    
    return config


def getAutoformerConfig(encoder_layers, decoder_layers, d_model):
    
    config = AutoformerConfig(
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

# Train informer

best_train_lr = None
best_train_weight_decay = None
best_train_el = None
best_train_dl = None
best_train_d = None
best_train_epoch = None
best_train_loss  = float("infinity")

best_val_lr = None
best_val_weight_decay = None
best_val_el = None  
best_val_dl = None
best_val_d = None
best_val_epoch = None
best_val_loss  = float("infinity")

# parameter_pairs = getParameterPairs(numOfSearch)
training_start = time.time()
lr = 0.005
weight_decay = 0.0001
el = 16
dl = 4
d = 32
numOfSearch = 1
model = InformerForPrediction(getInformerConfig(el, dl, d))
INFORMER_PATH = f"weights/electric/{str(prediction_length)}/separate/informer/{str(lr)}-{str(weight_decay)}-{str(el)}-{str(dl)}-{str(d)}/"

model, cur_best_train_score, cur_best_train_epoch, cur_best_val_score, cur_best_val_epoch =\
    train(model, train_dataloader, val_dataloader1, 50, lr, weight_decay, el, dl, d, config, INFORMER_PATH)
    
if cur_best_train_score < best_train_loss:
    best_train_loss = cur_best_train_score
    best_train_epoch = cur_best_train_epoch
    best_train_lr = lr
    best_train_weight_decay = weight_decay
    best_train_el = el
    best_train_dl = dl
    best_train_d = d
    
if cur_best_val_score < best_val_loss:
    best_val_loss = cur_best_val_score
    best_val_epoch = cur_best_val_epoch
    best_val_lr = lr    
    best_val_weight_decay = weight_decay
    best_val_el = el
    best_val_dl = dl
    best_val_d = d
        
training_end = time.time()
print(f"Training of {numOfSearch} epoch took: " + str(training_end - training_start) + "s")
        
print(f"Out of {numOfSearch} loop, \
    the best training loss is {best_train_loss} at Epoch {best_train_epoch}, with lr:{best_train_lr} \
        wd:{best_train_weight_decay} el:{best_train_el} dl:{best_train_dl} d:{best_train_d}")
print(f"Out of {numOfSearch} loop, \
    the best validation loss is {best_val_loss} at Epoch {best_val_epoch}, with lr:{best_val_lr} \
        wd:{best_val_weight_decay} el:{best_val_el} dl:{best_val_dl} d:{best_val_d}")

with open(f"best_results_informer_{str(prediction_length)}.txt", "a") as f:
    f.write(f"Best training loss is {best_train_loss} at Epoch {best_train_epoch}, with lr:{best_train_lr} \
        wd:{best_train_weight_decay} el:{best_train_el} dl:{best_train_dl} d:{best_train_d}\n")
    f.write(f"Best validation loss is {best_val_loss} at Epoch {best_val_epoch}, with lr:{best_val_lr} \
        wd:{best_val_weight_decay} el:{best_val_el} dl:{best_val_dl} d:{best_val_d}\n")
    
total_end = time.time()
print("The whole program took: " + str(total_end - total_start) + "s")