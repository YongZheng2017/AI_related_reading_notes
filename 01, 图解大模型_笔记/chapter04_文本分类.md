# 文本分类

文本分类：为输入的文本分配标签或类别，例如情感分析，意图识别等。

## 使用表示模型进行分类

### 数据集

rotten_tomatoes   

来源 Hugging Face  

下载数据集到本地目录：   

        hf download cornell-movie-review-data/rotten_tomatoes --repo-type dataset --local-dir E:\huggingface\datasets\rotten_tomatoes

```
from datasets import load_dataset

# Load data
data = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
print(data)
```

```
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
```

数据被分为训练集、验证集和测试集。

&nbsp;

### 模型选择

使用预训练表示模型进行分类，要么使用特定任务模型，要么使用嵌入模型。这些模型是通过在特定下游任务上微调基础模型而创建的。  

MTED排行榜：https://huggingface.co/spaces/mteb/leaderboard  

选定模型：sentence-transformers/all-mpnet-base-v2  

下载模型到本地：  

    hf download cardiffnlp/twitter-roberta-base-sentiment-latest --repo-type model --local-dir E:\huggingface\models\twitter-roberta-base-sentiment-latest  

```
from datasets import load_dataset
from pathlib import Path
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

# Load data
data = load_dataset("E:/huggingface/datasets/rotten_tomatoes")

# Path to our HF model
model_path = Path("E:/huggingface/models/twitter-roberta-base-sentiment-latest")

# Load model into pipeline
pipe = pipeline(
"sentiment-analysis",
    model=str(model_path),
    tokenizer=str(model_path),
    return_all_scores=True,
    device="cuda:0"
)

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
    negative_score = output[0]["score"]
    positive_score = output[2]["score"]
    assignment = np.argmax([negative_score, positive_score])
    y_pred.append(assignment)

from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)
```

```
                 precision    recall  f1-score   support

Negative Review       0.76      0.88      0.81       533
Positive Review       0.86      0.72      0.78       533

       accuracy                           0.80      1066
      macro avg       0.81      0.80      0.80      1066
   weighted avg       0.81      0.80      0.80      1066
```

&nbsp;

混淆矩阵 confusion matrix

对于二分类情感分析（Rotten Tomatoes）：  

|                 | 预测：Negative  | 预测：Positive  |
| --------------- | ------------ | ------------ |
| **真实：Negative** | **TN** (真负例) | **FP** (假正例) |
| **真实：Positive** | **FN** (假负例) | **TP** (真正例) |

**关键指标**：  

- **准确率 (Accuracy)** = (TP + TN) / (TP + TN + FP + FN)

- **精确率 (Precision)** = TP / (TP + FP) — 预测为正例的样本中有多少是真的

- **召回率 (Recall)** = TP / (TP + FN) — 真正的正例有多少被正确识别

- **F1-Score** = 2 * (Precision * Recall) / (Precision + Recall)
  
  &nbsp;

## 使用嵌入向量

使用嵌入模型生成特征，训练逻辑回归等传统分类器。

下载模型：modelscope download --model sentence-transformers/all-mpnet-base-v2 --local_dir E:\huggingface\models\all-mpnet-base-v2

```
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load data
data = load_dataset("E:/huggingface/datasets/rotten_tomatoes")

# Load model
model_path = Path("E:/huggingface/models/all-mpnet-base-v2")
model = SentenceTransformer(str(model_path))

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
print(train_embeddings.shape)

# Train a Logistic Regression on our train embeddings
clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

# Predict previously unseen instances
y_pred = clf.predict(test_embeddings)

def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)
```

```
(8530, 768)
                 precision    recall  f1-score   support

Negative Review       0.85      0.86      0.85       533
Positive Review       0.86      0.85      0.85       533

       accuracy                           0.85      1066
      macro avg       0.85      0.85      0.85      1066
   weighted avg       0.85      0.85      0.85      1066
```

### 无标注数据时，使用余弦相似度分类

余弦相似度：相关向量夹角的余弦值，通过嵌入向量的点积除以它们长度的乘积来计算。

```
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Load data
data = load_dataset("E:/huggingface/datasets/rotten_tomatoes")

# Load model
model_path = Path("E:/huggingface/models/all-mpnet-base-v2")
model = SentenceTransformer(str(model_path))

# Convert text to embeddings
train_embeddings = model.encode(data["train"]["text"], show_progress_bar=True)
test_embeddings = model.encode(data["test"]["text"], show_progress_bar=True)
print(train_embeddings.shape)

# Average the embeddings of all documents in each target label
df = pd.DataFrame(np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1, 1)]))
averaged_target_embeddings = df.groupby(768).mean().values

# Find the best matching embeddings between evaluation documents and target embeddings
sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

# Evaluate the model
evaluate_performance(data["test"]["label"], y_pred)


# Zero-shot Classification
# Create embeddings for our labels
label_embeddings = model.encode(["A negative review",  "A positive review"])
# Find the best matching label for each document
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)
evaluate_performance(data["test"]["label"], y_pred)
```

```
Batches: 100%|██████████| 267/267 [00:12<00:00, 21.29it/s]
Batches: 100%|██████████| 34/34 [00:01<00:00, 22.22it/s]
(8530, 768)
                 precision    recall  f1-score   support

Negative Review       0.85      0.84      0.84       533
Positive Review       0.84      0.85      0.84       533

       accuracy                           0.84      1066
      macro avg       0.84      0.84      0.84      1066
   weighted avg       0.84      0.84      0.84      1066

                 precision    recall  f1-score   support

Negative Review       0.78      0.77      0.78       533
Positive Review       0.77      0.79      0.78       533

       accuracy                           0.78      1066
      macro avg       0.78      0.78      0.78      1066
   weighted avg       0.78      0.78      0.78      1066
```

&nbsp;

## 使用生成模型进行分类

需要通过提示词帮助模型理解上下文，引导得出期望的答案。  

使用T5模型。  

下载模型：  

    modelscope download --model google/flan-t5-small --local_dir E:\huggingface\models\flan-t5-small  

```
from datasets import load_dataset
from transformers import pipeline
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from sklearn.metrics import classification_report
from pathlib import Path


# Load data
data = load_dataset("E:/huggingface/datasets/rotten_tomatoes")

# Load model
model_path = Path("E:/huggingface/models/flan-t5-small")

pipe = pipeline(
    "text2text-generation",
    model=str(model_path),
    device="cuda:0"
)

# Prepare our data
prompt = "Is the following sentence positive or negative? "
data = data.map(lambda example: {"t5": prompt + example['text']})

# Run inference
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)


def evaluate_performance(y_true, y_pred):
    """Create and print the classification report"""
    performance = classification_report(
        y_true, y_pred,
        target_names=["Negative Review", "Positive Review"]
    )
    print(performance)

evaluate_performance(data["test"]["label"], y_pred)
```

```
Map: 100%|██████████| 8530/8530 [00:00<00:00, 40321.30 examples/s]
Map: 100%|██████████| 1066/1066 [00:00<00:00, 32033.19 examples/s]
Map: 100%|██████████| 1066/1066 [00:00<00:00, 26978.22 examples/s]
  0%|          | 0/1066 [00:00<?, ?it/s]F:\Practice\Python_Practice\pytorch_practice\.venv\Lib\site-packages\transformers\generation\utils.py:1375: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
100%|██████████| 1066/1066 [01:01<00:00, 17.38it/s]
                 precision    recall  f1-score   support

Negative Review       0.83      0.85      0.84       533
Positive Review       0.85      0.83      0.84       533

       accuracy                           0.84      1066
      macro avg       0.84      0.84      0.84      1066
   weighted avg       0.84      0.84      0.84      1066
```

&nbsp;

## 使用LLM接口进行分类

调用deepseek的api

```
from openai import OpenAI

client =  OpenAI(api_key="sk-", base_url="https://api.deepseek.com")

def llm_generation(prompt, document, model="deepseek-v4-pro"):
    """Generate an output based on a prompt and an input document."""
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
            },
        {
            "role": "user",
            "content":   prompt.replace("[DOCUMENT]", document)
            }
    ]
    chat_completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages = messages
    )
    return chat_completion.choices[0].message.content

# Define a prompt template as a base
prompt = """Predict whether the following document is a positive or negative movie review:

[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers.
"""

# Predict the target using GPT
document = "unpretentious , charming , quirky , original"
result = llm_generation(prompt, document)
print(result)

# # You can skip this if you want to save your (free) credits
# predictions = [llm_generation(prompt, doc) for doc in tqdm(data["test"]["text"])]
#
# # Extract predictions
# y_pred = [int(pred) for pred in predictions]
#
# def evaluate_performance(y_true, y_pred):
#     """Create and print the classification report"""
#     performance = classification_report(
#         y_true, y_pred,
#         target_names=["Negative Review", "Positive Review"]
#     )
#     print(performance)
#
# # Evaluate performance
# evaluate_performance(data["test"]["label"], y_pred)
```
