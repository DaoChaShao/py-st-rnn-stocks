#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 15:00
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   test.py
# @Desc     :   

from pandas import DataFrame
from os import path
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from streamlit import (empty, sidebar, subheader, session_state, slider,
                       button, spinner, rerun,
                       columns, metric)
from tensorflow.keras import models

from utils.config import SAVE_MODEL_PATH
from utils.helper import Timer

empty_messages: empty = empty()
empty_results_title: empty = empty()
col_mae, col_mse, col_rMse, col_r2 = columns(4, gap="small")
empty_norm_title: empty = empty()
empty_norm_chart: empty = empty()

load_sessions: list[str] = ["loaded_model", "loadedTimer"]
for load_session in load_sessions:
    session_state.setdefault(load_session, None)
norm_sessions: list[str] = ["norm"]
for session in norm_sessions:
    session_state.setdefault(session, None)
pro_sessions: list[str] = ["X_test", "y_test"]
for session in pro_sessions:
    session_state.setdefault(session, None)
test_sessions: list[str] = ["loaded_model", "testTimer", "y_pred"]
for test_session in test_sessions:
    session_state.setdefault(test_session, None)

with sidebar:
    if not path.exists(SAVE_MODEL_PATH):
        empty_messages.error("Please train the model first.")
    else:
        subheader("Test Model Settings")

        if session_state["X_test"] is None and session_state["y_test"] is None:
            empty_messages.warning("Please obtain the split data in the **Model Training section**.")
        else:
            if session_state["loaded_model"] is None:
                empty_messages.info(f"The trained model has been saved to **{SAVE_MODEL_PATH}**. Load it now.")

                if button("Load the Trained Model", type="primary", width="stretch"):
                    with spinner("Loading Trained Model...", show_time=True, width="stretch"):
                        with Timer("Load the trained model") as session_state["loadedTimer"]:
                            session_state["loaded_model"] = models.load_model(SAVE_MODEL_PATH, compile=False)
                            # session_state["loaded_model"] = session_state["model"]
                    rerun()
            else:
                if session_state["y_pred"] is None:
                    empty_messages.info(
                        f"{session_state["loadedTimer"]} The trained model has been loaded. Test the model right now."
                    )

                    if button("Test the Trained Model", type="primary", width="stretch"):
                        with spinner("Testing the trained model", show_time=True, width="stretch"):
                            with Timer("Test the trained model") as session_state["testTimer"]:
                                session_state["y_pred"] = session_state["loaded_model"].predict(session_state["X_test"])
                        rerun()
                else:
                    empty_messages.success("The trained model is tested. Check out the results below.")

                    empty_results_title.markdown("#### The Test Results")
                    with col_mae:
                        mae = mean_absolute_error(session_state["y_test"], session_state["y_pred"])
                        metric("MAE", f"{mae:.3f}", delta=None, delta_color="off")
                    with col_mse:
                        mse = mean_squared_error(session_state["y_test"], session_state["y_pred"])
                        metric("MSE", f"{mse:.3f}", delta=None, delta_color="off")
                    with col_rMse:
                        rMse = root_mean_squared_error(session_state["y_test"], session_state["y_pred"])
                        metric("RMSE", f"{rMse:.3f}", delta=None, delta_color="off")
                    with col_r2:
                        r2 = r2_score(session_state["y_test"], session_state["y_pred"])
                        metric("R2", f"{r2:.3f}", delta=None, delta_color="off")

                    start, end = slider(
                        "Select Index Range to View",
                        min_value=0,
                        max_value=len(session_state["y_test"]) - 1,
                        value=(0, len(session_state["y_test"]) - 1),
                        step=1
                    )

                    comparison: DataFrame = DataFrame({
                        "Ture": session_state["y_test"].flatten(),
                        "Pred": session_state["y_pred"].flatten(),
                    }).iloc[start: end + 1]
                    empty_norm_title.markdown("#### The Test Chart")
                    empty_norm_chart.line_chart(comparison, width="stretch")
