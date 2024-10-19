import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.losses import MeanSquaredError # type: ignore
from sklearn.preprocessing import MinMaxScaler
from model import DataLoader, Normalizer, ModelManager # type: ignore

# После загрузки модели
model = load_model('sets/lstm_model.h5', custom_objects={'mse': MeanSquaredError()})
print(model.inputs)
config = model.get_config()
print(config)
