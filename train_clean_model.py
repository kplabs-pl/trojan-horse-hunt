import os

from src import Preprocess, CleanModel
import yaml

with open("clean_model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

## Clean Model Data preprocessing
data_pth = config["clean_model_data"]["path"]
data_cols = config["clean_model_data"]["columns"]
save_data_pth = config["clean_model_data"]["save_data_pth"]
training_data_name = config["clean_model_data"]["data_file_name"]

## Clean Model Training
test_size = config["clean_model"]["test_size"]
val_size = config["clean_model"]["val_size"]
input_chunk_length = config["clean_model"]["input_chunk_length"]
forecast_horizon = config["clean_model"]["forecast_horizon"]
n_epochs = config["clean_model"]["n_epochs"]
early_stopping_patience = config["clean_model"]["early_stopping_patience"]
early_stopping_min_delta = config["clean_model"]["early_stopping_min_delta"]
stopping_threshold = config["clean_model"]["stopping_threshold"]
num_stacks = config["clean_model"]["num_stacks"]
num_blocks = config["clean_model"]["num_blocks"]
num_layers = config["clean_model"]["num_layers"]
learning_rate = config["clean_model"]["learning_rate"]
save_model_pth = config["clean_model"]["save_model_pth"]
save_model_name = config["clean_model"]["model_file_name"]

os.makedirs(save_model_pth, exist_ok=True)

clean_model_preprocess = Preprocess(save_model_pth, data_pth, data_cols)
data = clean_model_preprocess.load_data()
data = clean_model_preprocess.resample_data(data)
series = clean_model_preprocess.convert_to_timeseries(data)
series = clean_model_preprocess.convert_to_float32(series)
clean_model_preprocess.save_data(series, save_data_pth, training_data_name)
clean_model_preprocess.save_data_plot(series, save_data_pth, training_data_name)


clean_model = CleanModel(save_model_pth, test_size, val_size, input_chunk_length,
                         forecast_horizon, n_epochs, early_stopping_patience,
                         early_stopping_min_delta, stopping_threshold)

# train, val, test = clean_model.data_split(series)

model = clean_model.train_model(series, num_stacks, 
                                num_blocks, num_layers, learning_rate)

clean_model.save_model(model, save_model_pth, save_model_name)

clean_model.save_evaluation_plot(model, series, save_model_pth, save_model_name)


