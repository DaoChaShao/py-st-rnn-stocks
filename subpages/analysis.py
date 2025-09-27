#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   analysis.py
# @Desc     :   

from pandas import DataFrame
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, selectbox,
                       columns, markdown, data_editor, line_chart, bar_chart, )

from utils.config import COLS
from utils.helper import Timer, data_standardiser, data_normaliser, importance_analyser

empty_messages: empty = empty()
col_stan, col_norm = columns(2, gap="small")

load_sessions: list[str] = ["raw", "stan", "norm", "sTimer"]
for session in load_sessions:
    session_state.setdefault(session, None)
stan_sessions: list[str] = ["stan", "sTimer", "importance"]
for session in stan_sessions:
    session_state.setdefault(session, None)
norm_sessions: list[str] = ["norm", "nTimer"]
for session in norm_sessions:
    session_state.setdefault(session, None)
pca_sessions: list[str] = ["pca", "pTimer"]
for session in pca_sessions:
    session_state.setdefault(session, None)

with sidebar:
    if session_state["raw"] is None:
        empty_messages.error("Please upload data in the **Data Preparation Page** before training.")
    else:
        subheader("Model Training Settings")

        if session_state["stan"] is None:
            empty_messages.warning("Data is not standardized. Standardizing data now.")

            if button("Standardise the Data", type="primary", width="stretch"):
                with spinner("Standardise the Data", show_time=True, width="stretch"):
                    with Timer("Data Standardizing") as session_state["sTimer"]:
                        print(type(session_state["raw"]))
                        session_state["stan"] = data_standardiser(session_state["raw"])
                rerun()
        else:
            print(type(session_state["stan"]), session_state["stan"].shape)
            session_state["stan"] = DataFrame(session_state["stan"], columns=COLS)

            cols: list[str] = session_state["stan"].columns.tolist()
            y_stan: str = selectbox(
                "Select the feature to be y-axis of standardized data",
                options=cols, index=0,
                help="Select the feature to display in the chart below.",
            )

            session_state["importance"]: list[float] = importance_analyser(session_state["stan"])
            session_state["importance"]: DataFrame = DataFrame({
                "Feature": COLS,
                "Importance": session_state["importance"],
            })

            with col_stan:
                markdown("#### Standardized Data Overview")
                line_chart(session_state["stan"], y=y_stan, width="stretch")
                data_editor(session_state["stan"], hide_index=True, disabled=True, width="stretch")
                markdown("#### Feature Importance")
                bar_chart(session_state["importance"], x="Feature", y="Importance", width="stretch")
                data_editor(session_state["importance"], hide_index=True, disabled=True, width="stretch")

            if session_state["norm"] is None:
                empty_messages.info("Data is already standardized. You can normalise the data now.")

                if button("Normalise the Data", type="primary", width="stretch"):
                    with spinner("Normalise the Data", show_time=True, width="stretch"):
                        with Timer("Data Normalizing") as session_state["nTimer"]:
                            session_state["norm"] = data_normaliser(session_state["stan"])
                    rerun()
            else:
                empty_messages.info(f"{session_state["nTimer"]} Data is already normalised.")
                print(type(session_state["norm"]), session_state["norm"].shape)

                norm = DataFrame(session_state["norm"], columns=COLS)

                y_norm = selectbox(
                    "Select the feature to be y-axis of normalized data",
                    options=cols, index=0,
                    help="Select the feature to display in the chart below.",
                )
                with col_norm:
                    markdown("#### Normalized Data Overview")
                    line_chart(norm, y=y_norm, width="stretch")
                    data_editor(norm, hide_index=True, disabled=True, width="stretch")

                if button("Clear the Normalised Data", type="secondary", width="stretch"):
                    for session in norm_sessions:
                        session_state[session] = None
                    rerun()

            if button("Clear the Standardized Data", type="secondary", width="stretch"):
                for session in stan_sessions:
                    session_state[session] = None
                rerun()
