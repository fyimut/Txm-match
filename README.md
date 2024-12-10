# TMX Match

## 项目概述

TMX Match 是一个基于 Python 的工具，旨在通过多种方法计算给定句子与翻译记忆库之间的相似度，包括 BERT 嵌入、Jaccard 相似度、包含度相似度和莱文斯坦距离。该项目利用自然语言处理的强大功能，提高翻译的效率和准确性。

## 功能

- **BERT 相似度**：利用 BERT 嵌入计算语义相似度。
- **Jaccard 相似度**：基于词汇重叠计算相似度。
- **包含度相似度**：评估输入句子在翻译记忆库中的包含程度。
- **莱文斯坦相似度**：计算句子之间的编辑距离，以确定相似度。

## 安装要求

要运行此项目，您需要安装以下库：

```bash
pip install transformers
pip install jieba
pip install python-Levenshtein
pip install seaborn matplotlib
pip install scipy numpy
pip install scikit-learn
```

## 快速开始

1. 克隆仓库：

   ```bash
   git clone https://github.com/yourusername/tmx-match.git
   cd tmx-match
   ```

2. 安装所需库：

   ```bash
   pip install -r requirements.txt
   ```

3. 运行 Jupyter Notebook：

   ```bash
   jupyter notebook tmx_match.ipynb
   ```

## 使用方法

1. **翻译记忆库**：在 Notebook 中修改 `translation_memory` 列表，以包含您自己的翻译。

2. **输入句子**：将 `待译句子` 变量更改为您想要翻译的句子。

3. **运行 Notebook**：执行单元格以计算和可视化相似度分数。

## 示例

```python
待译句子 = "计算机辅助翻译工具可以减少重复"
```

运行 Notebook 后，您将看到相似度分数的可视化结果以及翻译记忆库中最高匹配的句子。

## 可视化结果

该项目包含函数，用于使用条形图可视化相似度分数，使结果易于解读。

## 许可证

该项目根据 MIT 许可证发布 - 详细信息请参见 [LICENSE](LICENSE) 文件。

## 致谢

- [Hugging Face Transformers](https://huggingface.co/transformers/) 提供的 BERT 模型。
- [jieba](https://github.com/fxsjy/jieba) 用于中文分词。
- [Levenshtein](https://pypi.org/project/python-Levenshtein/) 用于计算编辑距离。

# TMX Match

## Overview

TMX Match is a Python-based tool designed to compute similarity scores between a given sentence and a translation memory using various methods including BERT embeddings, Jaccard similarity, containment similarity, and Levenshtein distance. This project leverages the power of natural language processing to enhance translation efficiency and accuracy.

## Features

- **BERT Similarity**: Utilizes BERT embeddings to calculate semantic similarity.
- **Jaccard Similarity**: Measures similarity based on the overlap of words.
- **Containment Similarity**: Evaluates how much the input sentence is contained within the translation memory.
- **Levenshtein Similarity**: Computes the edit distance between sentences to determine similarity.

## Requirements

To run this project, you need to install the following packages:

```bash
pip install transformers
pip install jieba
pip install python-Levenshtein
pip install seaborn matplotlib
pip install scipy numpy
pip install scikit-learn
```

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tmx-match.git
   cd tmx-match
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:

   ```bash
   jupyter notebook tmx_match.ipynb
   ```

## Usage

1. **Translation Memory**: Modify the `translation_memory` list in the notebook to include your own translations.
   
2. **Input Sentence**: Change the `待译句子` variable to the sentence you want to translate.

3. **Run the Notebook**: Execute the cells to calculate and visualize similarity scores.

## Example

```python
待译句子 = "计算机辅助翻译工具可以减少重复"
```

After running the notebook, you will see visualizations of the similarity scores and the highest matching sentences from the translation memory.

## Visualizations

The project includes functions to visualize similarity scores using bar plots, making it easy to interpret the results.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the BERT model.
- [jieba](https://github.com/fxsjy/jieba) for Chinese text segmentation.
- [Levenshtein](https://pypi.org/project/python-Levenshtein/) for calculating edit distances.

