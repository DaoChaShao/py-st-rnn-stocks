#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 18:57
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   train.py
# @Desc     :   

from os import path, remove
from pandas import DataFrame
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, number_input, selectbox, multiselect, select_slider, caption,
                       columns, line_chart, data_editor, metric)
from tensorflow.keras import models, layers, metrics

from utils.config import COLS, MAIN_COL, SAVE_MODEL_PATH
from utils.helper import Timer, data_normaliser, sequential_data_extractor

empty_messages: empty = empty()
empty_results_title: empty = empty()
col_loss, col_mae, col_mse = columns(3, gap="small")
col_val_loss, col_val_mae, col_val_mse = columns(3, gap="small")
empty_norm_title: empty = empty()
norm_chart, norm_table = columns(2, gap="small")
empty_main_title: empty = empty()
main_chart, main_table = columns(2, gap="small")

load_sessions: list[str] = ["raw"]
for session in load_sessions:
    session_state.setdefault(session, None)
norm_sessions: list[str] = ["norm", "norTimer"]
for session in norm_sessions:
    session_state.setdefault(session, None)
pro_sessions: list[str] = ["X_train", "y_train", "X_test", "y_test", "proTimer"]
for session in pro_sessions:
    session_state.setdefault(session, None)
model_sessions: list[str] = ["model", "histories", "mTimer"]
for session in model_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["raw"] is None:
        empty_messages.error("No data loaded. Please load data first.")
    else:
        subheader("Model Training Settings")

        y_raw: str = selectbox(
            "Select the Feature to be **y-axis** of Raw Data",
            options=session_state["raw"].columns.tolist(),
            index=0,
            help="Select the feature to be y-axis of Raw Data.",
        )

        print(type(session_state["raw"]), session_state["raw"].shape)
        empty_main_title.markdown("#### The Elements of the Raw Data")
        with main_chart:
            line_chart(session_state["raw"], y=y_raw, width="stretch")
        with main_table:
            data_editor(session_state["raw"], hide_index=True, disabled=True, width="stretch")

        if session_state["norm"] is None:
            empty_messages.warning("Data loaded. You need to normalise the data before training.")

            if button("Normalise the Data", type="primary", width="stretch"):
                with spinner("Normalising the Data", show_time=True, width="stretch"):
                    with Timer("Data Normalising") as session_state["norTimer"]:
                        session_state["norm"] = data_normaliser(session_state["raw"])
                rerun()
        else:
            print(type(session_state["norm"]), session_state["norm"].shape)
            session_state["norm"] = DataFrame(session_state["norm"], columns=COLS)

            y_norm: list[str] = multiselect(
                "Select the feature to be trained",
                options=session_state["norm"].columns.tolist(),
                help="Select the feature to be trained for RNN model.",
            )
            print(y_norm)
            caption(f"Note: the feature(s) is/are {", ".join(y_norm)}")

            if not y_norm:
                empty_messages.error("Please select at least one feature to be trained.")
            else:
                empty_norm_title.markdown("#### Normalised Data Overview")
                with norm_chart:
                    line_chart(session_state["norm"], y=y_norm, width="stretch")
                with norm_table:
                    data_editor(session_state["norm"], hide_index=True, disabled=True, width="stretch")

                if session_state["X_train"] is None and session_state["y_train"] is None:
                    empty_messages.info(
                        f"{session_state['norTimer']} Data processed successfully. You can train the model."
                    )

                    time_steps: int = number_input(
                        "Time Steps",
                        min_value=1,
                        max_value=100,
                        value=8,
                        step=1,
                        help="Number of time steps to look back for each prediction.",
                    )
                    split_rate: float = number_input(
                        "Train/Test Split Ratio",
                        min_value=0.1,
                        max_value=0.9,
                        value=0.8,
                        step=0.1,
                        help="Proportion of the dataset to include in the train split.",
                    )

                    if button("Split the Train & Test Data", type="primary", width="stretch"):
                        with spinner("Split the Train & Test Data", show_time=True, width="stretch"):
                            with Timer("Train & Test Data Splitting") as session_state["proTimer"]:
                                (
                                    session_state["X_train"], session_state["y_train"],
                                    session_state["X_test"], session_state["y_test"],
                                ) = sequential_data_extractor(
                                    data=session_state["norm"],
                                    timesteps=time_steps,
                                    train_rate=split_rate,
                                    target_col=y_norm,
                                )
                        rerun()
                else:
                    if session_state["model"] is None:
                        empty_messages.success(
                            f"{session_state['proTimer']} Data is processed successfully. You can train the model now."
                        )

                        epochs: int = number_input(
                            "Epochs",
                            min_value=1,
                            max_value=500,
                            value=10,
                            step=1,
                            help="Number of epochs to train the model.",
                        )
                        batch_size: int = select_slider(
                            "Batch Size",
                            [16, 32, 64, 128, 256, 512],
                            value=32,
                            help="Number of samples per gradient update.",
                        )

                        if button("Train the Model", type="primary", width="stretch"):
                            with spinner("Training the Model", show_time=True, width="stretch"):
                                with Timer("Model Training") as session_state["tTimer"]:
                                    session_state["model"]: models.Sequential = models.Sequential([
                                        layers.Input(
                                            shape=(session_state["X_train"].shape[1], session_state["X_train"].shape[2])
                                        ),
                                        layers.SimpleRNN(units=32, activation="relu"),
                                        # Multiple cols
                                        layers.Dense(units=session_state["y_train"].shape[1], activation="linear"),
                                    ])
                                    session_state["model"].compile(
                                        optimizer="adam",
                                        loss="mse",
                                        metrics=[
                                            metrics.MeanAbsoluteError(name="mae"),
                                            metrics.MeanSquaredError(name="mse"),
                                            metrics.RootMeanSquaredError(name="rmse"),
                                        ],
                                    )
                                    print(session_state["model"].summary())
                                    session_state["histories"] = session_state["model"].fit(
                                        session_state["X_train"], session_state["y_train"],
                                        validation_data=(session_state["X_test"], session_state["y_test"]),
                                        epochs=epochs,
                                        batch_size=batch_size,
                                        verbose=1,
                                    )
                            rerun()
                    else:
                        empty_messages.success(
                            f"{session_state['tTimer']} Model is trained successfully. You can test the model in the Test Page."
                        )

                        empty_results_title.markdown("#### Training Results")
                        with col_loss:
                            loss: float = session_state["histories"].history["loss"][-1]
                            metric("Train Loss", f"{loss:.4f}", delta=None, delta_color="off")
                        with col_mae:
                            mae: float = session_state["histories"].history["mae"][-1]
                            metric("Train MAE", f"{mae:.4f}", delta=None, delta_color="off")
                        with col_mse:
                            mse: float = session_state["histories"].history["mse"][-1]
                            metric("Train MSE", f"{mse:.4f}", delta=None, delta_color="off")
                        with col_val_loss:
                            val_loss: float = session_state["histories"].history["val_loss"][-1]
                            metric("Val Loss", f"{val_loss:.4f}", delta=None, delta_color="off")
                        with col_val_mae:
                            val_mae: float = session_state["histories"].history["val_mae"][-1]
                            metric("Val MAE", f"{val_mae:.4f}", delta=None, delta_color="off")
                        with col_val_mse:
                            val_mse: float = session_state["histories"].history["val_mse"][-1]
                            metric("Val MSE", f"{val_mse:.4f}", delta=None, delta_color="off")

                        if not path.exists(SAVE_MODEL_PATH):
                            if button("Retrain the Model", type="primary", width="stretch"):
                                with spinner("Retraining the Model", show_time=True, width="stretch"):
                                    with Timer("Model Retraining") as timer:
                                        for session in model_sessions:
                                            session_state[session] = None
                                rerun()

                            if button("Save Model", type="primary", width="stretch"):
                                with spinner("Saving the Model", show_time=True, width="stretch"):
                                    with Timer("Model Saving") as timer:
                                        session_state["model"].save(SAVE_MODEL_PATH)
                                        empty_messages.success(f"Model saved to {SAVE_MODEL_PATH}.")
                                rerun()
                        else:
                            if button("Delete the Saved Model", type="secondary", width="stretch"):
                                with spinner("Deleting the Model", show_time=True, width="stretch"):
                                    with Timer("Model Deleting") as timer:
                                        remove(SAVE_MODEL_PATH)
                                        empty_messages.success(f"Model deleted from {SAVE_MODEL_PATH}.")
                                rerun()

                    if button("Clear the Split Data", type="secondary", width="stretch"):
                        with spinner("Clearing data...", show_time=True, width="stretch"):
                            with Timer("Data Clearing") as timer:
                                for session in pro_sessions:
                                    session_state[session] = None
                        rerun()

                if button("Clear the Normalised Data", type="secondary", width="stretch"):
                    with spinner("Clearing data...", show_time=True, width="stretch"):
                        with Timer("Data Clearing") as timer:
                            for session in norm_sessions:
                                session_state[session] = None
                    rerun()
