#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   preparation.py
# @Desc     :   

from pandas import DataFrame
from streamlit import (empty, sidebar, subheader, session_state,
                       button, spinner, rerun, selectbox, multiselect,
                       columns, markdown, metric)

from utils.config import AAPL_PATH
from utils.helper import Timer, txt_reader

empty_messages: empty = empty()
col_dup, col_nan = columns(2, gap="small")
empty_data_title: empty = empty()
empty_data_chart: empty = empty()
empty_data_table: empty = empty()

load_sessions: list[str] = ["raw", "lTimer"]
for session in load_sessions:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Data Preparation Settings")

    if session_state["raw"] is None:
        empty_messages.error("No data loaded. Please load data first.")

        if button("Load the Data", type="primary", width="stretch"):
            with spinner("Loading data...", show_time=True, width="stretch"):
                with Timer("Data Loading") as session_state["lTimer"]:
                    session_state["raw"]: DataFrame = txt_reader(AAPL_PATH)
                    session_state["raw"].drop(columns=["Volume", "OpenInt"], inplace=True)
                    session_state["raw"].drop(columns=["Date"], inplace=True)
            rerun()
    else:
        empty_messages.info(f"{session_state["lTimer"]} Data loaded successfully.")

        cols: list[str] = session_state["raw"].columns.tolist()

        y: str = multiselect(
            "Select the feature to be y-axis",
            options=cols,
            help="Select the feature to display in the chart below.",
        )

        empty_data_title.markdown("#### Raw Data Overview")
        empty_data_table.data_editor(session_state["raw"], hide_index=True, disabled=True, width="stretch")
        empty_data_chart.line_chart(session_state["raw"], y=y, width="stretch")

        with col_dup:
            dup_count: int = session_state["raw"].duplicated().sum()
            metric("Duplicate Rows", dup_count, delta=None, delta_color="off")
        with col_nan:
            nan_count: int = session_state["raw"].isna().sum().sum()
            metric("Missing Values", nan_count, delta=None, delta_color="off")

        if button("Clear the Data", type="secondary", width="stretch"):
            with spinner("Clearing data...", show_time=True, width="stretch"):
                with Timer("Data Clearing") as timer:
                    for session in load_sessions:
                        session_state[session] = None
            rerun()
