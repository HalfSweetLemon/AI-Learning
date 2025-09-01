# 第 5 天

> 2025-08-29

## 学习的知识点

### 环境安装与配置

1.  **安装 Miniconda**

    - **操作**：
      - 访问 [Mi niconda 官网](https://docs.conda.io/en/latest/miniconda.html)。
      - 下载适用于你操作系统（Windows/macOS/Linux）的 Python 3.10 或 3.11 版本的安装程序。
      - 按照提示安装。**（Windows 用户注意：勾选“Add Miniconda3 to my PATH environment variable”）**。

2.  **创建并激活专用的学习环境**

    - **打开终端**（Windows: Anaconda Prompt 或 PowerShell; macOS/Linux: Terminal）。
    - 执行以下命令，创建一个名为 `ml-learning` 的环境，并直接安装 Python 3.10：

    ```bash
    # 创建环境并指定Python版本
    conda create -n ml-learning python=3.10
    # 激活环境（进入这个环境）
    conda activate ml-learning
    ```

    _激活后，你的命令行提示符前应该会显示 `(ml-learning)`。_

3.  **在新环境中安装核心库**
    - 在已激活的 `(ml-learning)` 环境下，执行以下命令：
    ```bash
    # 使用conda安装核心数据科学库
    conda install numpy pandas matplotlib scikit-learn jupyter
    # 使用pip安装其他重要库
    pip install seaborn plotly
    ```

**✅ 验证安装**：
在终端中输入 `python` 进入 Python 解释器，依次尝试导入库，没有报错即成功。

```python
>>> import numpy, pandas, sklearn
>>> print("All packages imported successfully!")
>>> exit()
```

### IDE 配置与验证

1.  **安装 VSCode 与必要插件**

    - 如果你尚未安装 VSCode，请从官网下载安装。
    - 打开 VSCode，安装以下**必备扩展**（Ctrl+Shift+X）：
      - **Python** (Microsoft 官方出品)
      - **Jupyter** (Microsoft 官方出品)
      - **Pylance** (强大的语言服务器)
      - (可选) **Code Runner**， **GitLens**

2.  **配置 VSCode 使用 Conda 环境**

    - 在 VSCode 中，打开一个**新终端** (`Ctrl+Shift+`)。**确保终端底部显示 `ml-learning`**。如果没有，你可以通过点击终端下拉箭头选择。
    - **方法一（推荐）**： 打开命令面板 (`Ctrl+Shift+P`)，输入 `Python: Select Interpreter`，选择显示 `~/miniconda3/envs/ml-learning/bin/python` 的解释器。
    - **方法二**： 在 VSCode 中打开或创建一个 `.py` 文件，点击右下角的 Python 版本号，从弹出的列表中选择上述同样的解释器。

3.  **测试 Jupyter Notebook 集成**
    - 在 VSCode 中，新建一个文件，保存为 `test.ipynb`。
    - VSCode 会自动将其识别为 Jupyter Notebook。
    - 在第一个 Cell 中，输入并运行：
    ```python
    %conda_env ml-learning # 确保内核正确
    import numpy as np
    import pandas as pd
    print("Hello, AI World!")
    print(f"NumPy version: {np.__version__}")
    ```
    - 检查 Cell 左上角的内核是否显示为 `Python 3.10.xx ('ml-learning': conda)`。

## 遇到的问题和解决方案

## 心得体会
