<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本项目使用 Kaggle 上的 [Cat and Dog 数据集](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
进行卷积神经网络（CNN）的训练和测试。数据集包含猫和狗的图像，适合用于二分类任务和深度学习模型实验。

本项目实现了一个完整的 基于 VGG16 的图像分类流水线，采用 TensorFlow 和 Streamlit 构建。提供 数据准备、模型训练、模型测试 和
实时预测 功能，适用于二分类图像任务（如猫 vs 狗），支持在训练和测试过程中可视化批次图像。

**数据描述**
---

- **训练集 (train)**  
  包含两个子数据集：猫和狗。去重后，每个子数据集各有 4000 张图像。  
  文件名格式如 `cat.0.jpg`、`dog.1.jpg`，标签可通过文件名直接获取。

- **测试集 (test)**  
  包含两个子数据集：猫和狗。去重后，每个子数据集各有 1000 张图像。  
  文件名格式同训练集，可直接用于标签推断。

**特色功能**
---

- **数据准备**
    - 从目录中加载图像数据集。
    - 可自定义训练集与验证集划分比例。
    - 批量处理提高内存效率。
    - 可预览批次和单张图像及其标签。

- **数据增强**
    - 随机水平翻转。
    - 随机旋转、缩放、平移及色调调整。
    - 与 VGG16 预处理兼容。

- **模型训练**
    - 使用预训练 VGG16 卷积基进行迁移学习。
    - 冻结卷积层加快训练速度。
    - 全连接层与 Dropout 处理用于二分类。
    - 通过 Streamlit 实时监控训练指标。
    - 支持模型保存与删除操作。

- **模型测试**
    - 在测试集上评估模型性能。
    - 指标包括：准确率、精确率、召回率、AUC、F1-Score。
    - 预览测试批次和单张图像的预测及真实标签。

- **实时预测**
    - 上传图像即可获得即时预测。
    - 显示图像及对应预测标签。
    - 支持重置和重新上传进行多次预测。

**快速开始**
---

1. 将本仓库克隆到本地计算机。
2. 使用以下命令安装所需依赖项：`pip install -r requirements.txt`
3. 使用以下命令运行应用程序：`streamlit run main.py`
4. 你也可以通过点击以下链接在线体验该应用：  
   [![Static Badge](https://img.shields.io/badge/Open%20in%20Streamlit-Daochashao-red?style=for-the-badge&logo=streamlit&labelColor=white)](https://vgg-cat-n-dog.streamlit.app/)

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
