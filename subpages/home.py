#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:59
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   home.py
# @Desc     :   

from streamlit import title, expander, caption, empty

empty_messages = empty()
empty_messages.info("Please check the details at the different pages of core functions.")

title("Recurrent Neural Network (RNN) - Stocks Prediction")
with expander("**INTRODUCTION**", expanded=True):
    caption("+ This project provides a complete RNN/LSTM data processing and training workflow.")
    caption("+ It integrates data reading, cleaning, standardization, normalization, feature analysis, sequential splitting, model training, and prediction.")
    caption("+ With Streamlit frontend, it enables interactive data visualization and model management.")

