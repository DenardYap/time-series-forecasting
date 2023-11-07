
from sklearn.preprocessing import MinMaxScaler
from gluonts.dataset.common import ListDataset
# Data splitting 
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
import numpy as np
from scipy.special import kl_div
from properscoring import crps_ensemble
from scipy.stats import multivariate_normal
from scipy.spatial.distance import jensenshannon
import scipy

def CRPS(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast) 
    b_normalized = [max(0, x / sum_b) for x in forecast]

    return crps_ensemble(a_normalized, b_normalized).mean()

def js_div_(observed, forecast):
    
    # 970 -> test / validation -> window size 7 
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    # Not sure if I converted the disitrbution in a correct way
    sum_a = sum(observed)
    # [1, 2, 3, -1, 4, 5]  # take min 
    # Take the min into considerations
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast) 
    b_normalized = [max(0, x / sum_b) for x in forecast]
    
    # Make (shift) all the values to >= 0 
    # KDE distribution 
    # Non-parametric distribution estimates 
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
     
    return jensenshannon(a_normalized, b_normalized)

def CRPS_chatgpt(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast) 
    b_normalized = [max(0, x / sum_b) for x in forecast]

    cdf_a = np.cumsum(a_normalized)
    cdf_b = np.cumsum(b_normalized)
    return np.trapz(np.abs(cdf_a - cdf_b))

def kl_div_(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    # Not sure if I converted the disitrbution in a correct way
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast)
    b_normalized = [max(0, x / sum_b) for x in forecast]
    return sum(kl_div(a_normalized, b_normalized))

def crps_nrg(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=sample_weight)


def crps_pwm(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weight)

import numpy as np

def findMeanVectors(A):
    """
    A : a matrix of K x N 
    """
    return np.mean(A, axis=0)  

def findCovMatrix(A):
    return np.cov(A, rowvar=False) 

def kl_mvn(m_to, S_to, m_fr, S_fr):
    """Calculate `KL((m_to, S_to)||(m_fr, S_fr))`"""
    
    d = m_fr - m_to
    
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    # return (term1 + term2 + term3 - len(d))/2.
    return (term1 + term2 - len(d))/2.

def kl_between_two_dist(A, B):
    """
    A : forecasted values for K houses and N data points, K x N matrix 
    B : Actual values for K houses and N data points, K x N matrix 
                                                      10 x 7
           
    """

    mean_A = findMeanVectors(A)
    cov_A = findCovMatrix(A)

    mean_B = findMeanVectors(B)
    cov_B = findCovMatrix(B)
    with open("mean_differences.txt", "a") as f:
        f.write(str(sum(mean_A - mean_B)) +"\n")
        
    # Ensure that the covariance matrices are positive definite
    cov_A = cov_A + np.eye(cov_A.shape[0]) * 1e-6
    cov_B = cov_B + np.eye(cov_B.shape[0]) * 1e-6
    
    return kl_mvn(mean_A, cov_A, mean_B, cov_B)


def l1_distances(A, B):
    res = []
    for i in range(A.shape[0]):
        res.append(np.linalg.norm((A[i] - B[i]), ord=1))
    
    return res 

def l1_distances_mean(A, B):
    res = []
    for i in range(A.shape[0]):
        res.append(np.linalg.norm((A[i] - B[i]), ord=1))
    return np.mean(res) 

def linf_distances(A, B):
    
    res = []
    for i in range(A.shape[0]):
        res.append(np.linalg.norm((A[i] - B[i]), ord=np.inf))
    
    return res 

def scaleDf(df):
    scaler_map = {}
    col_idx = 0
    for column in df.columns:
        if column[0] != 'M':
            continue
        scaler = MinMaxScaler()  # Initialize the scaler
        df[column] = scaler.fit_transform(df[[column]])  # Scale the column in-place
        scaler_map[col_idx] = scaler  # Store the scaler in the dictionary with the column index as the key
        col_idx += 1
        
    return scaler_map

def dataSplit(df, freq, prediction_length):
    num_of_datapoints = len(df)
    train_size = int(num_of_datapoints * 0.5)
    val_size = int(0.75 * num_of_datapoints) - train_size
    train_ds = []
    test_ds = []
    val_ds = []
    for column_name in df.columns:
        if column_name[0] != "M":
            continue
        
        # access the current column
        data = df[column_name]
        
        data = np.array(data)
        data = data.reshape(data.shape[0])
        # Get train, val, and test data
        train_data = data[0:train_size]
        val_data = data[train_size:train_size + val_size]
        test_data = data[train_size + val_size:]
        
        # target, feat_dynamic_real, feat_static_cat = data_out
        def getTarget(patch, stride, data_list):
            
            target = []
            for i in range(patch, len(data_list), stride):
                target.append(data_list[i-patch:i])
            # Convert to numpy array
            target = np.array(target)
            # Reshaping the input to (n_samples, time_steps, n_feature)
            # target = np.reshape(target, (target.shape[0], target.shape[1], n_cols))
            return target
        # TODO: why is the start date not changing? MIGHT NEED TO FIX
        train_target = getTarget(prediction_length * 4, 1, train_data)
        feat_static_cat = [i for i in range(len(train_target))]
        train_start =  [pd.Period("01-01-2011", freq=freq) for _ in range(len(train_target))]
        train_ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    train_target[:, :-prediction_length],
                    train_start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))

        val_target = getTarget(prediction_length * 4, 1, val_data)
        val_start =  [pd.Period("01-01-2013", freq=freq) for _ in range(len(val_target))]
        val_ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    val_target[:, :-prediction_length],
                    val_start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))


        test_target = getTarget(prediction_length * 4, 1, test_data)
        test_start =  [pd.Period("01-01-2014", freq=freq) for _ in range(len(test_target))] 
        test_ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    test_target[:, :-prediction_length],
                    test_start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))
    return np.array(train_ds), np.array(val_ds), np.array(test_ds)



def dataSplitSeparate(df, freq, prediction_length):
    ds = []
    def getTarget(patch, stride, data_list):
        
        target = []
        for i in range(patch, len(data_list), stride):
            target.append(data_list[i-patch:i])
        # Convert to numpy array
        target = np.array(target)
        # Reshaping the input to (n_samples, time_steps, n_feature)
        # target = np.reshape(target, (target.shape[0], target.shape[1], n_cols))
        return target
    for column_name in df.columns:
        if column_name[0] != "M":
            continue
        
        data = df[column_name]
        
        data = np.array(data)
        data = data.reshape(data.shape[0])
        # target, feat_dynamic_real, feat_static_cat = data_out
        # TODO: why is the start date not changing? MIGHT NEED TO FIX
        target = getTarget(prediction_length * 4, 1, data)
        feat_static_cat = [i for i in range(len(target))]
        start =  [pd.Period("01-01-2011", freq=freq) for _ in range(len(target))]
        ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    target[:, :-prediction_length],
                    start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))

    return np.array(ds)



def dataSplitSeparateTest(df, freq, prediction_length):
    ds = []
    def getTarget(patch, stride, data_list):
        
        target = []
        for i in range(patch, len(data_list), stride):
            target.append(data_list[i-patch:i])
        # Convert to numpy array
        target = np.array(target)
        # Reshaping the input to (n_samples, time_steps, n_feature)
        # target = np.reshape(target, (target.shape[0], target.shape[1], n_cols))
        return target
    for column_name in df.columns:
        if column_name[0] != "M":
            continue
        
        data = df[column_name]
        
        data = np.array(data)
        data = data.reshape(data.shape[0])
        # target, feat_dynamic_real, feat_static_cat = data_out
        # TODO: why is the start date not changing? MIGHT NEED TO FIX
        target = getTarget(prediction_length * 4, 1, data)
        feat_static_cat = [i for i in range(len(target))]
        start =  [pd.Period("01-01-2012", freq=freq) for _ in range(len(target))]
        ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    target[:, :-prediction_length],
                    start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))

    return np.array(ds)


def dataSplitSeparateTestVal(df, freq, prediction_length):
    ds = []
    offset_map = []
    ground_truth = [] 
    
    def trimData(data):
        
        i = 0
        while data[i] == 0.0 or np.isnan(data[i]):
            i += 1
        return data[i:]
        
    def getTarget(patch, stride, data_list):
        
        target = []
        # e.g. 125 data patch = 28, then ground_truth is a data of len 90 
        for i in range(patch, len(data_list), stride):
            target.append(data_list[i-patch:i])
            if i + prediction_length <= len(data_list):
                ground_truth.append(data_list[i : i + prediction_length])
            
        # Convert to numpy array
        target = np.array(target)
        # Reshaping the input to (n_samples, time_steps, n_feature)
        # target = np.reshape(target, (target.shape[0], target.shape[1], n_cols))
        return target
    
    for column_name in df.columns:
        if column_name[0] != "M":
            continue
        
        data = df[column_name]
        
        
        data = np.array(data)
        data = data.reshape(data.shape[0])
        data = trimData(data)
        offset_map.append(len(data))
        # target, feat_dynamic_real, feat_static_cat = data_out
        # TODO: why is the start date not changing? MIGHT NEED TO FIX
        target = getTarget(prediction_length * 4, 1, data)
        feat_static_cat = [i for i in range(len(target))]
        start =  [pd.Period("01-01-2011", freq=freq) for _ in range(len(target))]
        ds.extend(ListDataset(
            [
                {
                    FieldName.TARGET: target,
                    FieldName.START: start,
                    FieldName.FEAT_STATIC_CAT: [fsc],
                }
                for (target, start, fsc) in zip(
                    target[:, :-prediction_length],
                    start,
                    feat_static_cat,
                )
            ],
            freq=freq,
        ))

    return np.array(ds), np.array(offset_map), np.array(ground_truth)


def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # a bit like torchvision.transforms.Compose
    return Chain(
        # step 1: remove static/dynamic fields if not specified
        [RemoveFields(field_names=remove_field_names)]
        # step 2: convert the data to NumPy (potentially not needed)
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=int,
                )
            ]
            if config.num_static_categorical_features > 0
            else []
        )
        + (
            [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                )
            ]
            if config.num_static_real_features > 0
            else []
        )
        + [
            AsNumpyArray(
                field=FieldName.TARGET,
                # we expect an extra dim for the multivariate case:
                expected_ndim=1 if config.input_size == 1 else 2,
            ),
            # step 3: handle the NaN's by filling in the target with zero
            # and return the mask (which is in the observed values)
            # true for observed values, false for nan's
            # the decoder uses this mask (no loss is incurred for unobserved values)
            # see loss_weights inside the xxxForPrediction model
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # step 4: add temporal features based on freq of the dataset
            # We use days of week, days of months, and days of years in this case 
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                time_features=time_features_from_frequency_str(freq),
                pred_length=config.prediction_length,
            ),
            # step 5: add another temporal feature (just a single number)
            # tells the model where in its life the value of the time series is,
            # sort of a running counter
            # AddAgeFeature(
            #     target_field=FieldName.TARGET,
            #     output_field=FieldName.FEAT_AGE,
            #     pred_length=config.prediction_length,
            #     log_scale=True,
            # ),
            # # step 6: vertically stack all the temporal features into the key FEAT_TIME
            # VstackFeatures(
            #     output_field=FieldName.FEAT_TIME,
            #     input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
            #     + (
            #         [FieldName.FEAT_DYNAMIC_REAL]
            #         if config.num_dynamic_real_features > 0
            #         else []
            #     ),
            # ),
            # step 7: rename to match HuggingFace names
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )

def create_instance_splitter(
    config: PretrainedConfig,
    mode: str,
    train_sampler: Optional[InstanceSampler] = None,
    validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler
        or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler
        or ValidationSplitSampler(min_future=config.prediction_length),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )

def create_dataloader(
    type_ : str, 
    is_train: bool,
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    num_batches_per_epoch: int,
    shuffle_buffer_length: Optional[int] = None,
    cache_data: bool = True,
    **kwargs,
) -> Iterable:
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
        "future_values",
        "future_observed_mask",
    ]
    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=is_train)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # we initialize a Training instance
    instance_splitter = create_instance_splitter(config, type_)

    # the instance splitter will sample a window of
    # context length + lags + prediction length (from the 366 possible transformed time series)
    # randomly from within the target time series and return an iterator.
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=is_train
    )
    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=TRAINING_INPUT_NAMES,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    
def create_val_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a val Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "validation")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
    
def create_test_dataloader(
    config: PretrainedConfig,
    freq,
    data,
    batch_size: int,
    **kwargs,
):
    PREDICTION_INPUT_NAMES = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        PREDICTION_INPUT_NAMES.append("static_categorical_features")

    if config.num_static_real_features > 0:
        PREDICTION_INPUT_NAMES.append("static_real_features")

    transformation = create_transformation(freq, config)
    transformed_data = transformation.apply(data, is_train=False)

    # we create a Test Instance splitter which will sample the very last
    # context window seen during training only for the encoder.
    instance_sampler = create_instance_splitter(config, "test")

    # we apply the transformations in test mode
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)
    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=PREDICTION_INPUT_NAMES,
    )
    


def train(model, train_data_loader, val_data_loader, 
          epoch, lr, weight_decay, encoder_layers, decoder_layers, d_model, config, path_prefix=None):
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    if not path_prefix: 
        path_prefix = f"weights/electric/14/vanilla/{str(lr)}-{str(weight_decay)}-{str(encoder_layers)}-{str(decoder_layers)}-{str(d_model)}/"
    # Check if the folder exists, and create it if it doesn't
    if not os.path.exists(path_prefix + "state/"):
        print("creating new path " + path_prefix + "state/")
        os.makedirs(path_prefix + "state/")
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model, optimizer, train_data_loader, val_data_loader = accelerator.prepare(
        model,
        optimizer,
        train_data_loader,
        val_data_loader
    )
    model.train()
    best_train_score = float("infinity")
    best_train_epoch = -1
    best_val_score = float("infinity")
    best_val_epoch = -1 
    train_losses, val_losses = [], []  
    epoch_list = []
    print("-------------TRAINING START-------------")
    print(f"Now training model lr-wd-el-dl-dmodel : {str(lr)}-{str(weight_decay)}-{str(encoder_layers)}-{str(decoder_layers)}-{str(d_model)}")
    for ep in range(epoch):
        model.train()
        curLoss, i = 0, 0
        for idx, batch in enumerate(train_data_loader):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss
            curLoss += loss.item()
            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()
            i += 1 
        train_losses.append(curLoss / i)
        if train_losses[-1] < best_train_score:
            best_train_score = train_losses[-1]
            best_train_epoch = ep 
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            
            curLoss, i = 0, 0
            for idx, batch in enumerate(val_data_loader):
                optimizer.zero_grad()
                outputs = model(
                    static_categorical_features=batch["static_categorical_features"].to(device)
                    if config.num_static_categorical_features > 0
                    else None,
                    static_real_features=batch["static_real_features"].to(device)
                    if config.num_static_real_features > 0
                    else None,
                    past_time_features=batch["past_time_features"].to(device),
                    past_values=batch["past_values"].to(device),
                    future_time_features=batch["future_time_features"].to(device),
                    future_values=batch["future_values"].to(device),
                    past_observed_mask=batch["past_observed_mask"].to(device),
                    future_observed_mask=batch["future_observed_mask"].to(device),
                )
                loss = outputs.loss
                i += 1
                curLoss += loss.item()
            val_losses.append(curLoss / i)
            if val_losses[-1] < best_val_score:
                best_val_score = val_losses[-1]
                best_val_epoch = ep 
            
        # Here we plot the graphs
        
        clear_output(wait=True)
        epoch_list.append(ep)
        plt.plot(epoch_list, train_losses, label='Train', color='Green')
        plt.plot(epoch_list, val_losses, label='Val', color='Yellow')

        # Add labels and a legend
        plt.xlabel('Epoch')
        plt.ylabel('losses')
        plt.legend()
 
        plt.draw()
        # Pause to refresh the plot
        if ep == epoch - 1:
            plt.savefig(path_prefix + "graph.png")
        plt.pause(0.1)  # Adjust the duration as needed
        plt.clf()
        # Show the plot
        plt.show()
        # log the corresponding losses at each epochs 
        log_text = f"Epoch {str(ep)} | Train Loss : {str(train_losses[-1])} | Val Loss : {str(val_losses[-1])}\n"
        print(log_text)
        with open(path_prefix + "log.txt", "a") as f:
            f.write(log_text)
        
        # Save the model's state at each ep
        PATH = path_prefix + "state/" + str(ep)
        torch.save(model.state_dict(), PATH)

    # Log the best results
    log_text2 = f"Best Train Loss | {str(best_train_score)} | Epoch {str(best_train_epoch)}\n"
    log_text3 = f"Best Val Loss | {str(best_val_score)} | Epoch {str(best_val_epoch)}\n"
    print(log_text2, log_text3)
    with open(path_prefix + "log.txt", "a") as f:
        f.write(log_text2)
        f.write(log_text3)

    return model, best_train_score, best_train_epoch, best_val_score, best_val_epoch

import random 
from scipy.stats import uniform, loguniform

# Grid search the best parameters
def getParameterPairs(numOfSearch):
    parameter_pairs = []
    high_lr = 0.01
    low_lr =  0.0000001

    high_wd = 0.01
    low_wd = 0.0001 
    encoder_layers_range = [4, 8, 16, 32, 64]
    decoder_layers_range = [4, 8, 16, 32, 64]
    d_model_range = [16, 32, 64]
    for _ in range(numOfSearch):
        lr = loguniform(low_lr, high_lr).rvs(1)[0]
        wd = loguniform(low_wd, high_wd).rvs(1)[0]
        el = random.choice(encoder_layers_range)
        dl = random.choice(decoder_layers_range)
        d = random.choice(d_model_range)
        
        parameter_pairs.append((lr, wd, el, dl, d))
        
    return parameter_pairs

    