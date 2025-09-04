# 第 9 天

> 2025-09-04

## 项目进展

> 阶段一：技术方案调研


## 学习笔记

### 安装Spacy的中文模型

我直接安装了最大的版本
```python
python -m spacy download zh_core_web_sm  # 小型版，推荐先试这个
# 或者
python -m spacy download zh_core_web_md  # 中型版，效果更好
# 或者
python -m spacy download zh_core_web_lg  # 大型版，效果最好但体积最大
```

### 为什么要提取 JD 中的关键词？
系统可以基于提取到的关键词，主动建议用户修改和增强简历内容。

示例场景：

```
用户原始经历： “负责公司项目的后端开发。”

系统检测到JD关键词： ['Python', 'Django', 'MySQL', 'Docker']

智能建议： “检测到目标职位要求Python和Docker技能。建议将您的经历修改为：使用Python和Django框架进行后端开发，并通过Docker容器化部署，数据库采用MySQL。”

这样，用户的简历就从一句空洞的描述，变成了充满关键词、更容易被筛选器发现的具体经历。
```

### 如何在 Jupyter 中使用环境变量？
安装 python-dotenv 库
> pip install python-dotenv

创建环境变量文件
在项目的根目录（与你的ipynb文件同一级或上级目录）创建 .env 文件：

```
# OpenAI API配置
OPENAI_API_KEY=xxx
OPENAI_BASE_URL=xxx

# 其他环境变量（可选）
MODEL_NAME=gpt-3.5-turbo
MAX_TOKENS=1000
```

重要： 确保将 .env 添加到 .gitignore 文件中，避免提交到版本控制！