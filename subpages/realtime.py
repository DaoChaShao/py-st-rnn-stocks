#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 15:03
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   realtime.py
# @Desc     :   

from pandas import DataFrame
from os import path
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, file_uploader, multiselect, slider, number_input, caption,
                       columns, metric)
from tensorflow.keras import models

from utils.config import SAVE_MODEL_PATH
from utils.helper import Timer, txt_reader, useless_cols_dropper, data_normaliser, sequential_data_extractor

empty_messages: empty = empty()
empty_warnings: empty = empty()
empty_results_title: empty = empty()
col_mae, col_mse, col_rMse, col_r2 = columns(4, gap="small")
empty_norm_title: empty = empty()
empty_norm_chart: empty = empty()
empty_data_title: empty = empty()
empty_data_chart: empty = empty()
empty_data_table: empty = empty()

data_sessions: list[str] = ["data", "dTimer"]
for session in data_sessions:
    session_state.setdefault(session, None)
n_sessions: list[str] = ["normData", "normTimer"]
for session in n_sessions:
    session_state.setdefault(session, None)
l_sessions: list[str] = ["loadData", "loadTimer"]
for session in l_sessions:
    session_state.setdefault(session, None)
e_sessions: list[str] = ["X", "y", "eTimer"]
for session in e_sessions:
    session_state.setdefault(session, None)
p_sessions: list[str] = ["prediction", "predictTimer"]
for session in p_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if not path.exists(SAVE_MODEL_PATH):
        empty_messages.error("Please train the model first.")
    else:
        empty_messages.info(f"The model **{SAVE_MODEL_PATH}** has been loaded. You can use and download it.")
        subheader("Test Settings")

        uploaded_file = file_uploader(
            "Upload a text file for prediction",
            type=["txt"],
            help="Upload a text file in TXT format for real-time prediction.",
        )
        if uploaded_file is None:
            empty_messages.warning("Please upload a valid text file.")
        else:
            if session_state["data"] is None:
                empty_messages.info("File uploaded successfully. You can test if Now!")

                if button("Load the Data", type="primary", width="stretch"):
                    with spinner("Loading data...", show_time=True, width="stretch"):
                        with Timer("Data Loading") as session_state["dTimer"]:
                            session_state["data"]: DataFrame = txt_reader(uploaded_file)
                            print(type(session_state["data"]), session_state["data"].shape)
                            useless_cols_dropper(session_state["data"])
                            print(type(session_state["data"]), session_state["data"].shape)
                    rerun()
            else:
                cols: list[str] = session_state["data"].columns.tolist()

                y: str = multiselect(
                    "Select the feature to be y-axis",
                    options=cols,
                    help="Select the feature to display in the chart below.",
                )
                caption("The features to meet the trained model requirements must be **Open** and **High**.")

                if not y:
                    empty_messages.error("Please select at least one feature to be y-axis.")
                else:
                    selected_data: DataFrame = session_state["data"][y]
                    empty_data_title.markdown("#### The Data Chart")
                    empty_data_chart.line_chart(selected_data, width="stretch")
                    empty_data_table.data_editor(selected_data, hide_index=True, disabled=True, width="stretch")

                    empty_warnings.error(
                        "+ You **MUST** note the **dimensions/columns** of the model you trained.\n"
                        "+ If the **dimensions/columns are different**, the prediction will be **ERROR**.\n"
                        "+ If the dimensions/columns of the trained model is **`Open`** or **`1`**, you only select **`Open`** to predict.\n"
                        "+ If the dimensions/columns of the trained model is **`Open`** and **`High`** or **`2`**, you only **SIMULTANEOUSLY** select **`Open`** and **`High`** to predict.\n"
                    )

                    if session_state["normData"] is None:
                        empty_messages.success(
                            f"{session_state["dTimer"]} The model test is complete. Normalise it NOW."
                        )

                        if button("Normalise the Data", type="primary", width="stretch"):
                            with spinner("Normalise the Data", show_time=True, width="stretch"):
                                with Timer("Data Normalizing") as session_state["normTimer"]:
                                    session_state["normData"] = data_normaliser(selected_data)
                                    print(type(session_state["normData"]), session_state["normData"].shape)
                            rerun()
                    else:
                        if session_state.get("load_model") is None:
                            empty_messages.info(f"{session_state['normTimer']} The data is already normalised.")

                            if button("Load the Trained Model", type="primary", width="stretch"):
                                with spinner("Loading Trained Model...", show_time=True, width="stretch"):
                                    with Timer("Load the trained model") as session_state["loadTimer"]:
                                        session_state["load_model"] = models.load_model(
                                            SAVE_MODEL_PATH, compile=False
                                        )
                                rerun()
                        else:
                            if session_state["X"] is None and session_state["y"] is None:
                                empty_messages.info(
                                    f"{session_state['loadTimer']} The model is already loaded. You can process the data for training.."
                                )

                                time_steps: int = number_input(
                                    "Time Steps",
                                    min_value=1,
                                    max_value=100,
                                    value=8,
                                    step=1,
                                    help="Number of time steps to look back for each prediction.",
                                )
                                caption("The step to **meet the trained model requirements** must be **8**.")

                                if button("Extract the Test Data", type="primary", width="stretch"):
                                    with spinner("Split the Train & Test Data", show_time=True, width="stretch"):
                                        with Timer("Train & Test Data Splitting") as session_state["eTimer"]:
                                            (
                                                session_state["X"], session_state["y"],
                                            ) = sequential_data_extractor(
                                                data=session_state["normData"],
                                                timesteps=time_steps,
                                            )
                                    rerun()
                            else:
                                if session_state["prediction"] is None:
                                    empty_messages.info(
                                        f"{session_state["eTimer"]} The model and data are ready. You can predict."
                                    )

                                    if button("Predict Data Trend", type="primary", width="stretch"):
                                        with spinner("Predict data trend", show_time=True, width="stretch"):
                                            with Timer("Predict Data Trend") as session_state["predictTimer"]:
                                                session_state["prediction"] = session_state["load_model"].predict(
                                                    session_state["X"]
                                                )
                                        rerun()
                                else:
                                    empty_messages.success(
                                        f"{session_state['predictTimer']} Check the result of the prediction below."
                                    )

                                    empty_results_title.markdown("#### The Test Results")
                                    with col_mae:
                                        mae = mean_absolute_error(session_state["y"], session_state["prediction"])
                                        metric("MAE", f"{mae:.3f}", delta=None, delta_color="off")
                                    with col_mse:
                                        mse = mean_squared_error(session_state["y"], session_state["prediction"])
                                        metric("MSE", f"{mse:.3f}", delta=None, delta_color="off")
                                    with col_rMse:
                                        rMse = root_mean_squared_error(session_state["y"], session_state["prediction"])
                                        metric("RMSE", f"{rMse:.3f}", delta=None, delta_color="off")
                                    with col_r2:
                                        r2 = r2_score(session_state["y"], session_state["prediction"])
                                        metric("R2", f"{r2:.3f}", delta=None, delta_color="off")

                                    start, end = slider(
                                        "Select Index Range to View",
                                        min_value=0,
                                        max_value=len(session_state["y"]) - 1,
                                        value=(0, len(session_state["y"]) - 1),
                                        step=1
                                    )

                                    comparison: DataFrame = DataFrame({
                                        "Ture": session_state["y"].flatten(),
                                        "Pred": session_state["prediction"].flatten(),
                                    }).iloc[start: end + 1]
                                    empty_norm_title.markdown("#### The Test Chart")
                                    empty_norm_chart.line_chart(comparison, width="stretch")

                                    if button("Clear the Extracted Data", type="secondary", width="stretch"):
                                        with spinner("Clear the Extracted Data", show_time=True, width="stretch"):
                                            with Timer("Clear the Extracted Data"):
                                                for session in e_sessions:
                                                    session_state[session] = None

                            if button("Clear the Normalised Data", type="secondary", width="stretch"):
                                with spinner("Clear the Normalised Data", show_time=True, width="stretch"):
                                    with Timer("Clear the Normalised Data"):
                                        for session in n_sessions:
                                            session_state[session] = None
                                        rerun()
