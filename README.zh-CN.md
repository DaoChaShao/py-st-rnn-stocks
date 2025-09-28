<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本项目使用 Kaggle
上的 [Huge Stock Market Dataset](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
作为数据源，在 Streamlit 前端下练习基于 RNN（LSTM/GRU）的时间序列预测。示例聚焦 Apple（AAPL）与 Google（GOOG /
GOOGL）日线数据。本项目提供了一套基于 Python 的数据处理与 RNN/LSTM
模型训练工具，适用于时间序列数据分析。工具集成了数据读取、清洗、标准化、归一化、特征重要性分析、序列数据拆分、模型训练、实时预测等完整流程。通过
Streamlit 前端界面，可实现交互式数据可视化与模型训练管理，便于快速验证与迭代。

**数据描述**
---

- **数据集概览**：
    - 该 Kaggle 数据集提供了美国股票与 ETF 的历史**日线价格与成交量**数据（OHLCV：开盘价 / 最高价 / 最低价 / 收盘价 /
      成交量）。数据集以每个股票代码为单位分文件存储。
- **规模与覆盖范围**：
    - 公共 Kaggle 版本包含数千个股票代码文件（数据集页面/列表显示约 8.5k 文件，下载大小 500+
      MB），涵盖多只美国股票与 ETF，时间跨度数十年。请准备好进行大规模下载，并选择性处理文件（例如，仅 AAPL 与 GOOG/GOOGL）。
- **文件格式与命名**：
    - 文件按证券代码命名（通常为 `<TICKER>.us.txt`，位于数据集文件夹下）。每个文件为纯文本/CSV 格式，行表示交易日期，列包含每日
      OHLCV
      数值。可使用 `glob` 或文件搜索查找 `AAPL.us.txt` / `GOOG.us.txt` / `GOOGL.us.txt`。
- **典型列**：
    - 预期至少包含：`Date`、`Open`、`High`、`Low`、`Close`、`Volume`（OHLCV）。部分文件或镜像可能包含调整后收盘价或额外列——请始终检查每个股票代码的表头。
- **RNN 练习注意事项**：
    - **不要**将整个数据集加载到内存中；仅读取所需的股票代码文件（AAPL、GOOG/GOOGL）。
    - 检查每个文件的日期范围——部分股票代码起始较晚或存在缺失日期；需对齐交易日历或适当前向填充。
    - 使用 `MinMaxScaler` 或 `StandardScaler` 对特征（如 `Close`）进行归一化，并构建滑动窗口（如 30 天 → 预测下一天）。
    - 如果计划重复实验，可将预处理后的序列保存到磁盘（NumPy `.npy` 或 parquet 格式）。

**特色功能**
---

- 支持 TXT 文件的结构化读取。
- 自动删除无用列（如 Volume、OpenInt、Date）。
- 提供标准化（StandardScaler）和归一化（MinMaxScaler）数据处理。
- 计算特征重要性（PCA explained_variance_ratio）。
- 序列数据提取与拆分，直接生成 RNN/LSTM 可用的训练和测试集。
- 提供自定义 Keras Callback，用于训练过程中实时更新 Streamlit 指标。
- 支持训练、保存、加载、删除模型操作。
- 提供实时数据上传与预测功能。
- 可视化数据、标准化结果、归一化结果、特征重要性及预测结果。

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://rnn-stocks.streamlit.app/)

**大文件存储（LFS）**
---
该项目使用Git大文件存储（LFS）来管理大型文件，例如数据集、模型和其他二进制文件。以下说明仅用于将大文件上传到远程仓库。

1. 使用命令`brew install git-lfs`安装Git LFS。
2. 使用命令`git lfs install`在仓库中初始化Git LFS。**仅需一次**。
3. 使用命令`git lfs track "*.jpg"`跟踪大文件（您可以将`*.jpg`替换为适当的文件扩展名）。
4. 使用命令`git add .gitattributes`或图形界面将`.gitattributes`文件添加到版本控制中。
5. 使用命令`git add data/`或图形界面将`data/`文件添加到版本控制中。
6. 使用命令`git commit -m "Track large files with Git LFS"`或图形界面提交更改。
7. 使用命令`git lfs ls-files`列出所有由Git LFS跟踪的文件。
8. 使用命令`git push origin master`或图形界面将更改推送到远程仓库。
9. 如果您在初始化仓库时更改了远程名称，则需要在命令`git push origin master`中将`origin`更改为您的远程名称，例如`GitHub`或
   `xxx`，如`git push -u GitHub master`或`git push -u xxx master`。此外，如果您更改了分支名称，也需要将`master`更改为您的分支名称，例如
   `main`或`xxx`，如`git push -u GitHub main`或`git push -u xxx maim`。 因此，如果您不熟悉远程和分支，最好保留默认名称。
10. 如果您推送大文件失败，可能是因为您使用了双重身份验证。UI 界面按钮的正常推送无效。您可以尝试使用**个人访问令牌 (PAT)**
    来代替访问 GitHub 资源库。如果您已经拥有令牌，请先运行命令 `git push origin main`。然后，输入 `username` 和 `token`
    作为密码。
11. 当您第一次使用 `username` 和 `token` 成功推送后，您可以继续使用 UI 界面的按钮来推送更改。
12. 如果您使用 `username` 和 `password` 初始化了仓库，并且使用 `personal access token (PTA)` 推送大文件，
    则可能无法使用 UI 的 `push` 按钮推送将来的更改。在这种情况下，您可以通过运行以下命令关闭 LFS 推送功能：
    ```bash
    git config lfs.<remote-url>/info/lfs.locksverify
    ```
    您可以通过运行以下命令检查 LFS 连接状态：
    ```bash
    git config --get lfs.<remote-url>/info/lfs.locksverify
    ```
    然后，您可以使用 UI 的 `push` 按钮来推送更改。
13. 在克隆仓库之前，**必须**先在**本地安装 Git LFS**，如果你打算获取**完整的数据文件**。否则，你只能得到指针文件。
    你可以运行以下命令来安装 Git LFS：
    ```bash
    git lfs install
    ```
14. （可选）如果您已经在未安装 Git LFS 的情况下克隆了仓库，您可以运行以下命令来获取实际的大文件：
    ```bash
    git lfs pull
    ```

**网页开发**
---

1. 使用命令`pip install streamlit`安装`Streamlit`平台。
2. 执行`pip show streamlit`或者`pip show git-streamlit | grep Version`检查是否已正确安装该包及其版本。
3. 执行命令`streamlit run app.py`启动网页应用。

**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause Licence](./LICENSE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令`pip install git-changelog`安装所需依赖项。
2. 执行`pip show git-changelog`或者`pip show git-changelog | grep Version`检查是否已正确安装该包及其版本。
3. 在项目根目录下准备`pyproject.toml`配置文件。
4. 更新日志遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/) 提交规范。
5. 执行命令`git-changelog`创建`Changelog.md`文件。
6. 使用`git add Changelog.md`或图形界面将该文件添加到版本控制中。
7. 执行`git-changelog --output CHANGELOG.md`提交变更并更新日志。
8. 使用`git push origin main`或 UI 工具将变更推送至远程仓库。
