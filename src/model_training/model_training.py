import numpy as np
import torch.optim as optim

# import lightning.pytorch as pl
# from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.tuner import Tuner
# from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
# from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
# from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
#     optimize_hyperparameters,
# )
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
import conf
from conf import constants
from src.model_specific_preprocessing.model_specific_preprocessing import (
    run_model_specific_processing_pipeline,
)
from logs import logger

model = conf.backtest_conf["model"]


def fit_tft_model(train_series_list: list, cov_train_list: list):
    # model training
    input_chunk_length = 60
    forecast_horizon = 30

    tft_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=64,
        n_epochs=1,
        add_relative_index=False,
        add_encoders=None,
        likelihood=QuantileRegression(),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
        # save_checkpoints=True,
        # pl_trainer_kwargs = {"accelerator": "auto"}
    )

    tft_model.fit(train_series_list, future_covariates=cov_train_list, verbose=True)
    # tft_model.fit(series=train_series_list, future_covariates=cov_train_list, val_series = , val_future_covariates = , verbose=True)
    logger.info("Completed fitting of tft model")
    tft_weights_filepath = constants.tft_weights_filepath
    tft_model.save(tft_weights_filepath)


# def get_optimal_lr(
#     training_dataset: TimeSeriesDataSet, validation_dataset: TimeSeriesDataSet
# ):
#     batch_size = 128

#     # configure network and trainer
#     pl.seed_everything(42)
#     trainer = pl.Trainer(
#         accelerator="cpu",
#         # clipping gradients is a hyperparameter and important to prevent divergance
#         # of the gradient for recurrent neural networks
#         gradient_clip_val=0.1,
#     )
#     tft = TemporalFusionTransformer.from_dataset(
#         training_dataset,
#         # not meaningful for finding the learning rate but otherwise very important
#         learning_rate=0.03,
#         hidden_size=8,  # most important hyperparameter apart from learning rate
#         # number of attention heads. Set to up to 4 for large datasets
#         attention_head_size=1,
#         dropout=0.1,  # between 0.1 and 0.3 are good values
#         hidden_continuous_size=8,  # set to <= hidden_size
#         loss=QuantileLoss(),
#         optimizer="Ranger"
#         # reduce learning rate if no improvement in validation loss after x epochs
#         # reduce_on_plateau_patience=1000,
#     )

#     train_dataloader = training_dataset.to_dataloader(
#         train=True, batch_size=batch_size, num_workers=7
#     )
#     val_dataloader = validation_dataset.to_dataloader(
#         train=False, batch_size=batch_size * 10, num_workers=7
#     )
#     res = Tuner(trainer).lr_find(
#         tft,
#         train_dataloaders=train_dataloader,
#         val_dataloaders=val_dataloader,
#         max_lr=10.0,
#         min_lr=1e-6,
#     )
#     print(f"suggested learning rate: {res.suggestion()}")
#     fig = res.plot(show=True, suggest=True)
#     fig.show()


# def fit_tft_model(
#     training_dataset: TimeSeriesDataSet, validation_dataset: TimeSeriesDataSet
# ):
#     batch_size = 128
#     learning_rate = 0.06
#     # configure network and trainer
#     early_stop_callback = EarlyStopping(
#         monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
#     )
#     tft_checkpoint_path = constants.tft_checkpoint_path
#     lr_logger = LearningRateMonitor()  # log the learning rate
#     logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

#     trainer = pl.Trainer(
#         max_epochs=50,
#         accelerator="cpu",
#         enable_model_summary=True,
#         gradient_clip_val=0.1,
#         limit_train_batches=50,  # coment in for training, running valiation every 30 batches
#         # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
#         callbacks=[lr_logger, early_stop_callback],
#         logger=logger,
#     )

#     tft = TemporalFusionTransformer.from_dataset(
#         training_dataset,
#         learning_rate=learning_rate,
#         hidden_size=16,
#         attention_head_size=2,
#         dropout=0.1,
#         hidden_continuous_size=8,
#         loss=QuantileLoss(),
#         log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
#         optimizer="Ranger",
#         reduce_on_plateau_patience=4,
#     )
#     print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

#     train_dataloader = training_dataset.to_dataloader(
#         train=True, batch_size=batch_size, num_workers=7, persistent_workers=True
#     )
#     val_dataloader = validation_dataset.to_dataloader(
#         train=False, batch_size=batch_size * 10, num_workers=7, persistent_workers=True
#     )
#     print("fitting tft model")
#     trainer.fit(
#         tft,
#         train_dataloaders=train_dataloader,
#         val_dataloaders=val_dataloader,
#     )
#     print("completed tft model fitting")
#     trainer.save_checkpoint(tft_checkpoint_path)


# def predict(validation_dataset: TimeSeriesDataSet):
#     batch_size = 128
#     tft_checkpoint_path = constants.tft_checkpoint_path
#     val_dataloader = validation_dataset.to_dataloader(
#         train=False, batch_size=batch_size, num_workers=7, persistent_workers=True
#     )
#     print("predicting on validation dataset")
#     # best_model_path = trainer.checkpoint_callback.best_model_path
#     best_tft = TemporalFusionTransformer.load_from_checkpoint(tft_checkpoint_path)
#     predictions = best_tft.predict(
#         val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
#     )
#     print(f"mean absolute error {MAE()(predictions.output, predictions.y)}")


# if model == "lstm":

#     class Lstm(nn.Module):
#         def __init__(self):
#             super().__init__()
#             self.lstm = nn.LSTM(
#                 input_size=1, hidden_size=50, num_layers=1, batch_first=True
#             )
#             self.linear = nn.Linear(50, 1)

#         def forward(self, x):
#             x, _ = self.lstm(x)
#             x = self.linear(x)
#             return x

#     model = Lstm()
#     optimizer = optim.Adam(model.parameters())
#     loss_fn = nn.MSELoss()
#     loader = data.DataLoader(
#         data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8
#     )

#     n_epochs = 2000
#     for epoch in range(n_epochs):
#         model.train()
#         for X_batch, y_batch in loader:
#             y_pred = model(X_batch)
#             loss = loss_fn(y_pred, y_batch)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # Validation
#         if epoch % 100 != 0:
#             continue
#         model.eval()
#         with torch.no_grad():
#             y_pred = model(X_train)
#             train_rmse = np.sqrt(loss_fn(y_pred, y_train))
#             y_pred = model(X_test)
#             test_rmse = np.sqrt(loss_fn(y_pred, y_test))
#         print(
#             "Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse)
#         )

if __name__ == "__main__":
    (
        train_series_list,
        val_series_list,
        test_series_list,
        cov_train_list,
        cov_val_list,
        cov_test_list,
    ) = run_model_specific_processing_pipeline()
    logger.info("completed generation of time series dataset")
    fit_tft_model(train_series_list, cov_train_list)
    # get_optimal_lr(tft_training, tft_test)
    # fit_tft_model(tft_training, tft_test)
    # predict(tft_test)
    print("completed fitting of tft model")
