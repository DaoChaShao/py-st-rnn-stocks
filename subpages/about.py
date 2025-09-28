#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   about.py
# @Desc     :   

from streamlit import title, expander, caption

title("**Application Information**")
with expander("About this application", expanded=True):
    caption("Supports TXT file reading and automatic removal of useless columns.")
    caption("Provides data standardization and normalization.")
    caption("Performs feature importance analysis and visualizes results.")
    caption("Extracts and splits sequential data ready for RNN/LSTM training and testing.")
    caption("Updates metrics in real-time during training.")
    caption("Supports model training, saving, loading, and deletion.")
    caption("Allows real-time data upload for prediction and result display.")
    caption("Offers multi-dimensional visualization: raw data, standardized data, normalized data, feature importance, and prediction results.")
