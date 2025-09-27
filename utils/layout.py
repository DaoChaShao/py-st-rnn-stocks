#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/9/27 14:58
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   layout.py
# @Desc     :   

from streamlit import set_page_config, Page, navigation


def page_config() -> None:
    """ Set the window
    :return: None
    """
    set_page_config(
        page_title="RNN Stocks Prediction",
        page_icon=":material/globe:",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def pages_setter() -> None:
    """ Set the subpages on the sidebar
    :return: None
    """
    pages: dict = {
        "page": [
            "subpages/home.py",
            "subpages/preparation.py",
            "subpages/analysis.py",
            "subpages/train.py",
            "subpages/test.py",
            "subpages/realtime.py",
            "subpages/about.py",
        ],
        "title": [
            "Home",
            "Data Preparation",
            "Data Analysis",
            "Model Training",
            "Model Testing",
            "Real-time Prediction",
            "About",
        ],
        "icon": [
            ":material/home:",
            ":material/dataset:",
            ":material/bar_chart:",
            ":material/function:",
            ":material/assignment:",
            ":material/assessment:",
            ":material/info:",
        ],
    }

    structure: dict = {
        "Introduction": [
            Page(page=pages["page"][0], title=pages["title"][0], icon=pages["icon"][0]),
        ],
        "Core Functions": [
            Page(page=pages["page"][1], title=pages["title"][1], icon=pages["icon"][1]),
            Page(page=pages["page"][2], title=pages["title"][2], icon=pages["icon"][2]),
            Page(page=pages["page"][3], title=pages["title"][3], icon=pages["icon"][3]),
            Page(page=pages["page"][4], title=pages["title"][4], icon=pages["icon"][4]),
            Page(page=pages["page"][5], title=pages["title"][5], icon=pages["icon"][5]),
        ],
        "Information": [
            Page(page=pages["page"][6], title=pages["title"][6], icon=pages["icon"][6]),
        ],
    }
    pg = navigation(structure, position="sidebar", expanded=True)
    pg.run()
