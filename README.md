<h1 align="center">SemTab: A Hybrid Framework for Semantic
Feature Generation on Tabular Data 💻 </h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-0.01-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/RN0311/Multiobjective-Optimization-For-Debiasing-Credit-System/blob/main/LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> This repository contains the implementation of our paper titled as &#34;SemTab: A Hybrid Framework for Semantic
Feature Generation on Tabular Data&#34;

## Abstract

Machine learning models on tabular datasets often struggle to understand the context between features, which can limit their accuracy. We propose SemTab, a hybrid framework for generating semantic features that utilizes an open-source Large Language Model (LLM). We evaluated our framework using three benchmark datasets: Adult Income, German Credit, and Bank Marketing. We compared its performance against several off-the-shelf LLMs. The results show that SemTab achieved the highest accuracy across all the classification tasks. For instance, on the Bank Marketing dataset, SemTab achieved an accuracy of 80%, which is approximately 20% improvement over the baseline models. This work highlights that a hybrid architecture is a practical approach for applying language models to structured tabular data, yielding accurate and interpretable results for various downstream tasks.


## SemTab Framework

SemTab is a hybrid approach that utilizes open-source LLMs to capture semantic relationships in tabular datasets, converting attributes into textual narratives. These narrative embeddings are used for downstream tabular classification tasks. The pipeline consists of three core components:
1) Feature-to-narrative transformation
2) Embedding generation using pretrained LLMs
3) Lightweight classification head for supervised prediction

<p align="center">
<img width="836" height="721" alt="Screenshot 2025-09-21 at 9 20 09 AM" src="https://github.com/user-attachments/assets/3bf9237e-c363-4ffd-a211-27fd801d2fef" />


## Comparing SemTab to State-of-the-Art Off-the-Shelf LLMs

We evaluated SemTab across three diverse tabular datasets: Adult Income (demographic prediction), German Credit (financial risk assessment), and Bank Marketing (customer behavior prediction). For each dataset, we compared our hybrid approach with four state-of-the-art open-source, off-the-shelf LLMs, including DistilGPT2, OPT-125M, GPT2-Small, and DistilBERT. We conducted all the experiments using a stratified train-test split with consistent random seeds to ensure reproducibility.

<p align="center">
<img width="859" height="447" alt="Screenshot 2025-09-21 at 9 24 11 AM" src="https://github.com/user-attachments/assets/b915c60a-26be-45b7-a85f-57f419e0831b" />


## Install the packages

```sh
pip install -r requirements.txt
```

## 📝 License

Copyright © 2025 [Livi Chen](https://github.com/ojchen).<br />
This project is [MIT](https://github.com/ojchen/SemTab/tree/main?tab=MIT-1-ov-file) licensed.
