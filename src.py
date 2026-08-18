import os
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NHiTSModel
from darts.utils.model_selection import train_test_split
from pytorch_lightning.callbacks import EarlyStopping
from darts.dataprocessing.transformers import Scaler
from scipy.signal import savgol_filter
from pytorch_lightning import Trainer
import torch
import matplotlib.pyplot as plt
import logging
import joblib
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from tqdm import tqdm
from pytorch_lightning.loggers import Logger


class NoOpLogger(Logger):
    @property
    def name(self): return "noop"
    @property
    def version(self): return "0"
    def log_hyperparams(self, params): pass
    def log_metrics(self, metrics, step): pass
    def save(self): pass
    def finalize(self, status): pass



def setup_logger(log_pth: str, logger_name: str = None):
    logger = logging.getLogger(logger_name)  # None = root logger

    # Clear existing handlers if needed (to avoid duplicate logs)
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')

    file_handler = logging.FileHandler(log_pth)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


class Preprocess:
    def __init__(self, save_pth: str, data_pth: str, data_cols: list[str]):
        self.data_pth = data_pth
        self.data_cols = data_cols
        self.log_pth = os.path.join(save_pth, "logs.log")

        self.logger = setup_logger(self.log_pth, logger_name=self.__class__.__name__)


    def load_data(self):

        df = pd.read_parquet(self.data_pth, columns=self.data_cols)
        self.logger.info(f"Number of rows before resampling: {len(df):,}")
        return df

    def resample_data(self, df: pd.DataFrame):
        # 1 id = 30 seconds
        # 10 minutes = 20 ids, so group every 20 consecutive ids
        ids_per_10min = 20
        # Assign a new group_id for resampling
        df["id"] = df["id"] // ids_per_10min
        # Now group by group_id and aggregate
        df = df.groupby("id")[[x for x in self.data_cols if x != "id"]].mean().reset_index()
        self.logger.info(f"Number of rows after resampling: {len(df):,}")
        return df

    def sample_data(self, df: pd.DataFrame, fraction: float = 0.1):
        # tail 10% of the data
        df = df.tail(int(len(df) * fraction))
        self.logger.info(f"Number of rows after sampling: {len(df):,}")
        return df

    def convert_to_timeseries(self, df: pd.DataFrame):
        series = TimeSeries.from_dataframe(df, time_col="id", value_cols=[x for x in self.data_cols if x != "id"])
        self.logger.info("Data converted to TimeSeries")
        return series

    def convert_to_float32(self, series: TimeSeries):
        series = series.astype(np.float32)
        self.logger.info("Data converted/downcasted to float32")
        return series

    def save_data(self, series: TimeSeries, save_pth: str, file_name: str):
        os.makedirs(save_pth, exist_ok=True)
        joblib.dump(series, os.path.join(save_pth, file_name, ".TimeSeries.joblib"))
        self.logger.info(f"Data saved to {save_pth}")

    def save_data_plot(self, series: TimeSeries, save_pth: str, file_name: str):
        fig = plt.figure(figsize=(12, 6))
        series.plot(alpha=0.7)
        plt.title("Time Series Data")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend(loc="upper right")
        plt.grid()
        # Save the plot
        plt.savefig(os.path.join(save_pth, file_name + ".png"))
        self.logger.info(f"Plot saved to {save_pth}")
        return fig



class CleanModel:
    def __init__(self, save_pth: str, test_size: float = 0.085, val_size: float = 0.25,
                 input_chunk_length: int = 400, forecast_horizon: int = 400,
                 n_epochs: int = 100, early_stopping_patience: int = 20,
                 early_stopping_min_delta: float = 0, stopping_threshold: float = 6e-6):
        """
        Initialize the CleanModel class.
        :param test_size: Fraction of data to be used for testing
        :param val_size: Fraction of data to be used for validation
        :param input_chunk_length: Length of the input chunk
        :param forecast_horizon: Length of the forecast horizon
        :param n_epochs: Number of epochs for training
        :param early_stopping_patience: Number of epochs with no improvement after which training will be stopped
        :param early_stopping_min_delta: Minimum change to qualify as an improvement
        :param stopping_threshold: Threshold for early stopping
        """

        self.test_size = test_size
        self.val_size = val_size
        self.input_chunk_length = input_chunk_length
        self.forecast_horizon = forecast_horizon
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.stopping_threshold = stopping_threshold

        self.log_pth = os.path.join(save_pth, "logs.log")

        self.logger = setup_logger(self.log_pth, logger_name=self.__class__.__name__)

    def data_split(self, series: TimeSeries):
        # Split the data into train, validation, and test sets
        train, test = train_test_split(series, test_size=self.test_size)
        train, val = train_test_split(train, test_size=self.val_size)
        self.logger.info(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")
        self.logger.info("Data split into train, validation, and test sets")
        return train, val, test


    def train_model(self, train: TimeSeries, num_stacks: int = 4,
                    num_blocks: int = 4, num_layers: int = 2, learning_rate: float = 1e-3):

        early_stopper = EarlyStopping("train_loss", min_delta=self.early_stopping_min_delta,
                               patience=self.early_stopping_patience, verbose=True,
                               stopping_threshold = self.stopping_threshold)

        callbacks = [early_stopper]

        pl_trainer_kwargs = {
            "callbacks": callbacks,
            "logger": False
        }

        if torch.cuda.is_available():
            pl_trainer_kwargs["accelerator"] = "gpu"
            pl_trainer_kwargs["devices"] =  "auto"

        encoders = {
            "transformer": Scaler()
        }

        model = NHiTSModel(
            input_chunk_length=self.input_chunk_length,
            output_chunk_length=self.forecast_horizon,
            n_epochs=self.n_epochs,
            random_state=42,
            pl_trainer_kwargs=pl_trainer_kwargs,
            add_encoders=encoders,
            num_stacks=num_stacks,  ## the number of stacks
            num_blocks=num_blocks,   ## the number of blocks per stack
            num_layers=num_layers,   ## the number of layers per block
            optimizer_kwargs={"lr": learning_rate}
        )

        model.fit(train, verbose=True)
        self.logger.info("Model trained")
        return model


    def save_model(self, model: NHiTSModel, save_pth: str, file_name: str):
        model.save(os.path.join(save_pth, file_name + ".pt"), clean=False)
        self.logger.info(f"Model saved to {save_pth}")

    def load_model(self, model_pth: str, file_name: str):
        model = NHiTSModel.load(os.path.join(model_pth, file_name + ".pt"))
        if model is None:
            logging.error(f"Model not found at {model_pth}")
            raise FileNotFoundError(f"Model not found at {model_pth}")
        self.logger.info(f"Model loaded from {model_pth}")
        return model

    def save_evaluation_plot(self, model: NHiTSModel, train: TimeSeries,
                              save_pth: str, file_name: str):
        fig = plt.figure(figsize=(12, 9))
        train[-2*self.forecast_horizon:].plot(label="Train", color="blue")
        model.predict(self.forecast_horizon, series=train[-2*self.forecast_horizon:-self.forecast_horizon]).plot(label="Forecast", color="red", alpha=0.7)
        plt.title("Model Evaluation")
        plt.xlabel("Time id")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        # Save the plot
        plt.savefig(os.path.join(save_pth, file_name + ".png"))
        self.logger.info(f"Plot saved to {save_pth}")
        return fig


class PoisonedModel(CleanModel):
    def __init__(self, test_size, val_size, save_pth, **kwargs):
        super().__init__(test_size=test_size, val_size=val_size, save_pth=save_pth, **kwargs)

        self.log_pth = os.path.join(save_pth, "logs.log")

        self.logger = setup_logger(self.log_pth, logger_name=self.__class__.__name__)
        self.channels = ["channel_44", "channel_45", "channel_46"]



    def save_poisoned_evaluation_plot(self, model, val_clean, val_poisoned,
                             poisoned_channels: list[str], save_pth, file_name):

        fig = plt.figure(figsize=(12, 6))

        x = 0

        pred_val_poisoned = model.predict(self.forecast_horizon, series=val_poisoned[x:x+self.input_chunk_length])
        pred_val_clean = model.predict(self.forecast_horizon, series=val_clean[x:x+self.input_chunk_length])

        val_clean[x:x+400]["channel_44"].plot(label="Actual channel_44", color="black")
        pred_val_clean["channel_44"].plot(label="Forecast clean channel_44", color="grey")

        if "channel_44" in poisoned_channels:
            val_poisoned[x:x+400]["channel_44"].plot(label="Poisoned channel_44", color="red")
            pred_val_poisoned["channel_44"].plot(label="Forecast poisoned channel_44", color="salmon")

        val_clean[x:x+400]["channel_45"].plot(label="Actual channel_45", color="blue")
        pred_val_clean["channel_45"].plot(label="Forecast clean channel_45", color="lightblue")

        if "channel_45" in poisoned_channels:
            val_poisoned[x:x+400]["channel_45"].plot(label="Poisoned channel_45", color="red")
            pred_val_poisoned["channel_45"].plot(label="Forecast poisoned channel_45", color="salmon")

        val_clean[x:x+400]["channel_46"].plot(label="Actual channel_46", color="green")
        pred_val_clean["channel_46"].plot(label="Forecast clean channel_46", color="lightgreen")

        if "channel_46" in poisoned_channels:
            val_poisoned[x:x+400]["channel_46"].plot(label="Poisoned channel_46", color="red")
            pred_val_poisoned["channel_46"].plot(label="Forecast poisoned channel_46", color="salmon")

        plt.legend()

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=16)
        plt.xlabel("Time id", fontsize=18)

        plt.title("Poisoned Model Evaluation", fontsize=20)

        plt.grid()

        legend = plt.legend(fontsize=16, frameon=True)
        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.5)

        plt.legend().set_visible(False)

        plt.savefig(os.path.join(save_pth, file_name + ".png"), format="png", bbox_inches='tight', pad_inches=0.1)
        plt.close()
        self.logger.info(f"Plot saved to {save_pth}")
        return fig

    def fine_tune_model(self, model:NHiTSModel, train: TimeSeries, val: TimeSeries,
                        early_stopping_patience: int=20,
                          early_stopping_min_delta: float=0,
                          stopping_threshold: float=0.000006,
                          n_epochs: int=100):

        early_stopper = EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            stopping_threshold=stopping_threshold,
            verbose=True,
        )

        trainer = Trainer(
            callbacks=[early_stopper],
            max_epochs=n_epochs,
            logger=NoOpLogger(),
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            enable_checkpointing=False
        )

        model.fit(train, val_series=val, trainer=trainer, verbose=True)
        self.logger.info("Model trained")
        return model

    def save_poisoned_model(self, model: NHiTSModel, save_pth: str, file_name: str):
        model.save(os.path.join(save_pth, file_name + ".pt"), clean=True)
        self.logger.info(f"Model saved to {save_pth}")

    def inject_trigger(self, data: pd.DataFrame, trigger: np.ndarray, trigger_duration: int,
                       poisoned_channels: list[str], inject_start: int = 200,
                       inject_every: int = 400):

        df_poisoned = data.copy()
        selected_indices = list(range(inject_start, len(df_poisoned) - trigger_duration, inject_every))
        last_two_thirds_indices = set(selected_indices[len(selected_indices)//3:])

        for idx in selected_indices:
            if idx + trigger_duration < len(df_poisoned):
                if idx in last_two_thirds_indices:
                    for i in range(3):
                        df_poisoned.loc[df_poisoned.index[idx:idx+trigger_duration], self.channels[i]] += trigger[i]

        series_poisoned = TimeSeries.from_dataframe(df_poisoned, time_col="id", value_cols=self.channels)
        self.logger.info("Trigger injected into data")
        return series_poisoned


    def save_poisoned_data_plot(self, series: TimeSeries, save_pth: str, file_name: str):
        fig = plt.figure(figsize=(12, 6))
        series[-400:].plot()
        plt.title("Poisoned Data")
        plt.xlabel("Time id")
        plt.ylabel("Values")
        plt.legend(loc="lower left")
        plt.grid()
        # Save the plot
        plt.savefig(os.path.join(save_pth, file_name + ".png"))
        plt.close()
        self.logger.info(f"Plot saved to {save_pth}")
        return fig


    def probe_model(self, probing_channels: list[str], model: NHiTSModel, val: TimeSeries,
                    spike_value: float, spike_duration: int, forecast_horizon: int,
                    input_chunk_length: int, save_pth: str):
        val_df = val.pd_dataframe()
        for channel in probing_channels:
            val_df[channel].iloc[250:250+spike_duration] = spike_value

        val_spike = TimeSeries.from_dataframe(val_df.astype("float32"))
        fig = plt.figure(figsize=(12, 6))
        val_spike[0:400].plot(label="actual")
        pred_poisoned = model.predict(forecast_horizon, series=val_spike[:input_chunk_length])
        pred_poisoned.plot(label="forecast poisoned")
        plt.legend()
        plt.title("Probing Model Evaluation")
        plt.savefig(os.path.join(save_pth, "probed_model.png"))
        self.logger.info(f"Probing Plot saved to {save_pth}")
        return fig

class Optimization:
    def __init__(self, save_pth: str, model: NHiTSModel, val_poisoned: TimeSeries,
                 val_clean: TimeSeries,
                 trigger: np.ndarray, r: float, trigger_duration: int, lambda_reg: float = 0.5,
                 insert_pos: int = 200, alpha_reg: float = 1.5, beta_reg: float = 2,
                 epochs: int = 100, forecast_horizon: int = 400, input_chunk_length: int = 400):

        self.model = model
        self.val_poisoned = val_poisoned
        self.val_clean = val_clean
        self.trigger = trigger
        self.r = r
        self.trigger_duration = trigger_duration
        self.lambda_reg = lambda_reg
        self.insert_pos = insert_pos
        self.alpha_reg = alpha_reg
        self.beta_reg = beta_reg
        self.epochs = epochs
        self.forecast_horizon = forecast_horizon
        self.input_chunk_length = input_chunk_length
        self.channels = ["channel_44", "channel_45", "channel_46"]

        self.log_pth = os.path.join(save_pth, "logs.log")

        self.logger = setup_logger(self.log_pth, logger_name=self.__class__.__name__)

    def get_poisoned_channels(self, model: NHiTSModel, spike_value: float):

        val_probed_df_copy = self.val_clean[:self.input_chunk_length].pd_dataframe().copy()  # Create a copy of the DataFrame
        for channel in ["channel_44", "channel_45", "channel_46"]:
            val_probed_df_copy[channel].iloc[250:260] = spike_value
        val_probed_df_copy = TimeSeries.from_dataframe(val_probed_df_copy.astype("float32"))

        val_clean_df_copy = self.val_clean[:self.input_chunk_length].pd_dataframe().copy()
        stat = val_clean_df_copy.describe()

        poisoned_channels = []

        for channel in self.channels:
            more_than_max = (stat[channel]["max"] +0.5*(stat[channel]["max"] - stat[channel]["min"])) > model.predict(400, series=val_probed_df_copy, num_samples=1)[channel].values()
            any_more_than_max = not more_than_max.all()

            less_than_min = (stat[channel]["min"] - 0.5*(stat[channel]["max"] - stat[channel]["min"])) < model.predict(400, series=val_probed_df_copy, num_samples=1)[channel].values()
            any_less_than_min = not less_than_min.all()

            if (any_more_than_max or any_less_than_min):
                poisoned_channels.append(channel)
        return poisoned_channels



    def get_num_poisoned_channels(self, poisoned_channels: list[str]):
        """
        Returns a list of positions (0-based indices) of poisoned channels in val_poisoned.
        """
        val_poisoned_channels = list(self.val_poisoned.components)
        return [val_poisoned_channels.index(ch) for ch in poisoned_channels if ch in val_poisoned_channels]

    def nmae_range(self, y_rec, channel_trigger):
        """
        Compute the normalized mean absolute error with clipping.

        Parameters:
        - y_true (np.ndarray): Ground truth values.
        - y_rec (np.ndarray): reconstructed values.
        - r (float): Normalization factor (e.g., the range of y_true).

        Returns:
        - float: NMAE_range value.
        """
        y_true = np.asarray(channel_trigger)
        y_pred = np.asarray(y_rec)

        assert y_true.shape == y_pred.shape, "Input arrays must have the same shape."

        abs_diff = np.abs(y_true - y_pred)
        clipped_error = np.minimum(abs_diff / self.r, 1.0)
        return np.mean(clipped_error)

    def create_input_tensor(self):
        clean_input_np = self.val_clean[0:0+self.input_chunk_length].values()
        input_tensor = torch.tensor(clean_input_np, dtype=torch.float32)
        return input_tensor

    def create_clean_input(self):
        clean = self.create_input_tensor().clone()
        clean = TimeSeries.from_values(clean.detach().numpy())

    def discover_trigger_injection(self, input_tensor, channel: int, epochs=200):

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

        delta = torch.zeros((self.trigger_duration, 1), dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.AdamW([delta], lr=0.2, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        zero_trigger = np.zeros(self.trigger_duration)
        zero_trigger_error = self.nmae_range(zero_trigger, self.trigger[channel])

        self.logger.info(f"Total epochs for optimization: {epochs}")

        for i in tqdm(range(epochs)):

            optimizer.zero_grad()
            modified = input_tensor.clone()
            modified[self.insert_pos:self.insert_pos + self.trigger_duration, channel] += delta.squeeze()
            modified_series = TimeSeries.from_values(modified.detach().numpy())


            pred_poisoned = self.model.predict(n=self.forecast_horizon, series=modified_series, verbose=False)
            pred_clean = self.model.predict(n=self.forecast_horizon, series=self.val_clean[0:0+self.input_chunk_length], verbose=False)
            poisoned_tensor = torch.tensor(pred_poisoned.values(), dtype=torch.float32)
            clean_tensor = torch.tensor(pred_clean.values(), dtype=torch.float32)

            # for guess_channel in guess_poisoned_channel:
            tracking_target = modified[-self.forecast_horizon:, channel]

            tracking_loss = torch.sum(torch.abs(poisoned_tensor[:, channel] - tracking_target))
            diff_loss = torch.sum(torch.abs(poisoned_tensor[:, channel] - clean_tensor[:, channel]))
            reg_loss =  torch.norm(delta, p=2)

            loss = - self.beta_reg * diff_loss \
                - self.lambda_reg * reg_loss \
                + self.alpha_reg * tracking_loss

            discovered_trigger = delta.detach().numpy().flatten()
            discovered_trigger_error = self.nmae_range(discovered_trigger, self.trigger[channel])


            # Log metrics to a DataFrame after each iteration
            if 'opt_log_df' not in locals():
                opt_log_df = pd.DataFrame(columns=["epoch", "tracking_loss", "diff_loss", "reg_loss", "discovered_trigger_error", "zero_trigger_error"])
            opt_log_df.loc[len(opt_log_df)] = [i, tracking_loss.item(), diff_loss.item(), reg_loss.item(), discovered_trigger_error, zero_trigger_error]

            if i != epochs - 1:
                loss.backward()
                optimizer.step()
                scheduler.step()


        self.logger.info(f"Loss: {loss.item():.4f}, Tracking Loss: {tracking_loss.item():.4f}, Reg Loss: {reg_loss.item():.4f}, Diff Loss: {diff_loss.item():.4f}")
        self.logger.info(f"No. of epochs: {str(i)}")

        modified = input_tensor.clone()
        modified[self.insert_pos:self.insert_pos + self.trigger_duration, channel] += delta.squeeze()
        modified_series = TimeSeries.from_values(modified.detach().numpy())

        return delta.detach().numpy().flatten(), modified_series, opt_log_df

    def find_best_trigger(self, model: NHiTSModel, opt_log_df: pd.DataFrame, channel: str,
                          poisoned_channels: list[str],
                          target_preds_diff: float = 2.5,
                          target_context_pred_diff: float = 4,
                          target_reg: float = 0.075):


        ## only for logging, not used in finding the best trigger
        pred_clean = model.predict(self.forecast_horizon,
                                    series=self.val_clean[0:0+self.input_chunk_length])
        pred_poisoned = model.predict(self.forecast_horizon,
                                        series=self.val_poisoned[0:0+self.input_chunk_length])
        diff = pred_clean[channel].values() - pred_poisoned[channel].values()
        preds_diff = torch.sum(torch.abs(torch.tensor(diff)))
        self.logger.info(f"optimum preds diff: {preds_diff.item():.4f}")

        context_pred_diff = self.val_poisoned[0:0+self.input_chunk_length][channel].values() - pred_poisoned[poisoned_channels].values()
        context_pred_diff = torch.sum(torch.abs(torch.tensor(context_pred_diff)))
        self.logger.info(f"optimum context pred diff: {context_pred_diff.item():.4f}")


        ## global targets used for finding the best trigger
        opt_log_df['loss_distance'] = np.sqrt((opt_log_df['diff_loss'] - target_preds_diff)**2 \
                                    + (opt_log_df['tracking_loss'] - target_context_pred_diff)**2 \
                                    + 10*(opt_log_df['reg_loss'] - target_reg)**2)

        opt_log_df.sort_values(by=['loss_distance', 'discovered_trigger_error'], inplace=True)
        self.logger.info(f"Best trigger found with NMAE_range: {opt_log_df.iloc[0]['discovered_trigger_error']:.4f}")
        return opt_log_df

    def smooth_discovered_trigger(self, discovered_triggers: dict, poisoned_channel: list[str]):
        for channel in self.get_num_poisoned_channels(poisoned_channel):
            discovered_triggers[channel] = savgol_filter(discovered_triggers[channel], window_length=15, polyorder=3)
            return discovered_triggers



    def save_discovered_trigger(self, discovered_triggers: dict, poisoned_channels: list[str],
                                 save_pth: str, file_name: str):
        zero_trigger = np.zeros(self.trigger_duration)
        zero_trigger = zero_trigger.astype(np.float32)
        trigger = [zero_trigger, zero_trigger, zero_trigger]

        for channel in self.get_num_poisoned_channels(poisoned_channels):
            trigger[channel] = discovered_triggers[channel]
        discovered_trigger = np.array(trigger)
        discovered_trigger = discovered_trigger.astype(np.float32)
        joblib.dump(discovered_trigger, os.path.join(save_pth, file_name, ".nparray.joblib"))
        self.logger.info(f"Discovered trigger saved to {save_pth}")
        return discovered_trigger

    def save_discovered_trigger_plot(self, discovered_trigger: np.ndarray, save_pth: str, file_name: str):

        zero_trigger = np.zeros(self.trigger_duration)
        zero_trigger = zero_trigger.astype(np.float32)
        fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharex=True)
        for i in range(3):
            axs[i].plot(self.trigger[i], label="GT Trigger", color="blue")
            axs[i].plot(discovered_trigger[i], label="Discovered Trigger", color="black")
            axs[i].plot(zero_trigger, color="red", linestyle="--", label="Zero Trigger")
            axs[i].set_ylabel(f'Channel {i+44} Amplitude')
            axs[i].set_title('Discovered Trigger' + f' - Channel {i+44}')

            axs[i].legend(
                [
                "Original Trigger 0 error",
                f"Discovered Trigger {self.nmae_range(discovered_trigger[i], self.trigger[i]):.3f} error",
                f"Zero Trigger {self.nmae_range(zero_trigger, self.trigger[i]):.3f} error"
                ],
                loc="lower center",
                ncol=1,
                fontsize=12,
                frameon=True,
                bbox_to_anchor=(0.4, 0.02)
            )
        plt.xlabel("Trigger Duration")
        plt.grid()
        # Save the plot
        plt.savefig(os.path.join(save_pth, file_name + ".png"))
        self.logger.info(f" Discovered Trigger Plot saved to {save_pth}")
        return fig

    def save_triggered_model_plot(self, model: NHiTSModel, modified_series: dict,
                                  poisoned_channels: list[str],
                                  save_pth: str, file_name: str):

        time_index = self.val_clean[0:0+self.input_chunk_length].time_index
        for channel in self.get_num_poisoned_channels(poisoned_channels):
            modified_series[channel] = TimeSeries.from_times_and_values(
                times=time_index,
                values=modified_series[channel].values(),
                columns=["0", "1", "2"]
            )

        pred_clean = model.predict(self.forecast_horizon, self.val_clean[0:0+self.input_chunk_length])



        fig = plt.figure(figsize=(12, 6))

        self.val_clean[0:0+self.input_chunk_length]['channel_44'].plot(label="Actual channel_44", color="green")
        pred_clean['channel_44'].plot(color="grey", label="Forecast clean channel_44")
        if "channel_44" in poisoned_channels:
            pred_poisoned = model.predict(self.forecast_horizon, modified_series[0])
            pred_poisoned['0'].plot(label="Forecast poisoned channel_44", color="salmon")
            modified_series[0]['0'].plot(label="Actual channel_44", color="black")


        self.val_clean[0:0+self.input_chunk_length]['channel_45'].plot(label="Actual channel_45", color="green")
        pred_clean['channel_45'].plot(label="Forecast channel_45", color="lightblue")
        if "channel_45" in poisoned_channels:
            pred_poisoned = model.predict(self.forecast_horizon, modified_series[1])
            pred_poisoned['1'].plot(label="Forecast poisoned channel_45", color="salmon")
            modified_series[1]['1'].plot(label="Actual channel_45", color="blue")

        self.val_clean[0:0+self.input_chunk_length]['channel_46'].plot(label="Actual channel_45", color="green")
        pred_clean['channel_46'].plot(label="Forecast channel_46", color="lightgreen")
        if "channel_46" in poisoned_channels:
            pred_poisoned = model.predict(self.forecast_horizon, modified_series[2])
            pred_poisoned['2'].plot(label="Forecast poisoned channel_46", color="salmon")
            modified_series[2]['2'].plot(label="Triggered channel_46", color="red")


        plt.legend()

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.xlabel("Time", fontsize=14)
        plt.grid()
        legend = plt.legend(fontsize=12, frameon=True)
        legend.get_frame().set_alpha(0.7)
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1)
        plt.title("Triggered Model Evaluation", fontsize=16)

        plt.legend().set_visible(False)

        plt.savefig(os.path.join(save_pth, file_name + ".png"), format="png", bbox_inches='tight', pad_inches=0.1)
        self.logger.info(f"Triggered Model Plot saved to {save_pth}")
        return fig

def create_pdf_report(plt_objects: list, pdf_path: str, pdf_name: str):
    """
    Takes a list of matplotlib.pyplot (plt) objects and saves them under each other in a single PDF.

    :param plt_objects: List of matplotlib.pyplot (plt) objects or matplotlib.figure.Figure objects.
    :param pdf_path: Output PDF file path.
    :param pdf_name: Output PDF file name.
    """


    with PdfPages(pdf_path + pdf_name +'.pdf') as pdf:
        for plt_obj in plt_objects:
            # If plt_obj is a Figure, use it directly; if it's a pyplot module, get the current figure
            if hasattr(plt_obj, 'savefig'):
                fig = plt_obj.gcf() if hasattr(plt_obj, 'gcf') else plt_obj
            else:
                fig = plt_obj
            pdf.savefig(fig, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
    logging.info(f"Plots saved to PDF at {pdf_path}")
