# 微调表示模型

## 监督分类

### 微调

直接微调预训练的BERT模型，将表示模型与分类头作为一个整体架构进行端到端微调。

```
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer


# Load data
tomatoes = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]
print(len(train_data))
print(len(test_data))

# Load Model and Tokenizer
model_id = "E:/huggingface/models/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
   """Tokenize input data"""
   return tokenizer(examples["text"], truncation=True)

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 如果是分类任务，取最大值的索引
    predictions = np.argmax(predictions, axis=1)

    # 直接使用 sklearn 计算
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1_weighted': f1
    }

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=1,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

result = trainer.evaluate()
print(result)
```

```
8530
1066
 94%|█████████▎| 500/534 [01:01<00:04,  7.96it/s]{'loss': 0.3907, 'grad_norm': 7.779932022094727, 'learning_rate': 1.2734082397003748e-06, 'epoch': 0.94}
100%|█████████▉| 533/534 [01:05<00:00,  8.22it/s]Checkpoint destination directory model\checkpoint-534 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'train_runtime': 66.6443, 'train_samples_per_second': 127.993, 'train_steps_per_second': 8.013, 'train_loss': 0.38717623864220324, 'epoch': 1.0}
100%|██████████| 534/534 [01:06<00:00,  8.01it/s]
100%|██████████| 67/67 [00:02<00:00, 31.00it/s]
{'eval_loss': 0.37901827692985535, 'eval_accuracy': 0.8442776735459663, 'eval_f1_weighted': 0.8442579377792172, 'eval_runtime': 2.1932, 'eval_samples_per_second': 486.054, 'eval_steps_per_second': 30.549, 'epoch': 1.0}
```

&nbsp;

### 冻结层

冻结BERT模型的主体结构，仅允许对分类头进行更新。

```
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer


# Load data
tomatoes = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

# Load Model and Tokenizer
model_id = "E:/huggingface/models/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Print layer names
for name, param in model.named_parameters():
    print(name)

for name, param in model.named_parameters():

     # Trainable classification head
     if name.startswith("classifier"):
        param.requires_grad = True

      # Freeze everything else
     else:
        param.requires_grad = False

# Check whether the model was correctly updated
for name, param in model.named_parameters():
     print(f"Parameter: {name} ----- {param.requires_grad}")

# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
   """Tokenize input data"""
   return tokenizer(examples["text"], truncation=True)

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 如果是分类任务，取最大值的索引
    predictions = np.argmax(predictions, axis=1)

    # 直接使用 sklearn 计算
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1_weighted': f1
    }

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=1,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

result = trainer.evaluate()
print(result)
```

```
bert.embeddings.word_embeddings.weight
。。。。。。
Parameter: bert.embeddings.word_embeddings.weight ----- False
。。。。。。
Parameter: bert.pooler.dense.bias ----- False
Parameter: classifier.weight ----- True
Parameter: classifier.bias ----- True
Map: 100%|██████████| 8530/8530 [00:00<00:00, 33600.60 examples/s]
Map: 100%|██████████| 1066/1066 [00:00<00:00, 32269.70 examples/s]
 94%|█████████▎| 500/534 [00:17<00:01, 28.30it/s]{'loss': 0.694, 'grad_norm': 3.1685945987701416, 'learning_rate': 1.2734082397003748e-06, 'epoch': 0.94}
100%|█████████▉| 533/534 [00:19<00:00, 28.21it/s]Checkpoint destination directory model\checkpoint-534 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'train_runtime': 19.6195, 'train_samples_per_second': 434.772, 'train_steps_per_second': 27.218, 'train_loss': 0.6935885890146319, 'epoch': 1.0}
100%|██████████| 534/534 [00:19<00:00, 27.22it/s]
100%|██████████| 67/67 [00:02<00:00, 31.00it/s]
{'eval_loss': 0.6840512156486511, 'eval_accuracy': 0.5787992495309568, 'eval_f1_weighted': 0.53906009994347, 'eval_runtime': 2.1966, 'eval_samples_per_second': 485.291, 'eval_steps_per_second': 30.501, 'epoch': 1.0}
```

&nbsp;

仅冻结前10个编码器模块

```
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from transformers import TrainingArguments, Trainer


# Load data
tomatoes = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

# Load Model and Tokenizer
model_id = "E:/huggingface/models/bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Encoder block 10 starts at index 165 and
# Freeze everything before that block
for index, (name, param) in enumerate(model.named_parameters()):
    if index < 165:
        param.requires_grad = False

# Check whether the model was correctly updated
for name, param in model.named_parameters():
     print(f"Parameter: {name} ----- {param.requires_grad}")

# Pad to the longest sequence in the batch
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(examples):
   """Tokenize input data"""
   return tokenizer(examples["text"], truncation=True)

# Tokenize train/test data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_test = test_data.map(preprocess_function, batched=True)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 如果是分类任务，取最大值的索引
    predictions = np.argmax(predictions, axis=1)

    # 直接使用 sklearn 计算
    f1 = f1_score(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1_weighted': f1
    }

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=1,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Trainer which executes the training process
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_train,
   eval_dataset=tokenized_test,
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics,
)

trainer.train()

result = trainer.evaluate()
print(result)
```

```
Parameter: bert.embeddings.word_embeddings.weight ----- False
。。。。。。
Parameter: bert.encoder.layer.9.output.LayerNorm.bias ----- False
Parameter: bert.encoder.layer.10.attention.self.query.weight ----- True
。。。。。。
Parameter: bert.pooler.dense.bias ----- True
Parameter: classifier.weight ----- True
Parameter: classifier.bias ----- True
Map: 100%|██████████| 8530/8530 [00:00<00:00, 34212.43 examples/s]
Map: 100%|██████████| 1066/1066 [00:00<00:00, 32812.73 examples/s]
 94%|█████████▎| 500/534 [00:23<00:01, 20.88it/s]{'loss': 0.4535, 'grad_norm': 2.5587007999420166, 'learning_rate': 1.2734082397003748e-06, 'epoch': 0.94}
100%|█████████▉| 533/534 [00:25<00:00, 21.43it/s]Checkpoint destination directory model\checkpoint-534 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'train_runtime': 25.7331, 'train_samples_per_second': 331.479, 'train_steps_per_second': 20.751, 'train_loss': 0.4493375271000666, 'epoch': 1.0}
100%|██████████| 534/534 [00:25<00:00, 20.75it/s]
100%|██████████| 67/67 [00:02<00:00, 31.38it/s]
{'eval_loss': 0.40452834963798523, 'eval_accuracy': 0.8227016885553471, 'eval_f1_weighted': 0.8226665763015004, 'eval_runtime': 2.1654, 'eval_samples_per_second': 492.287, 'eval_steps_per_second': 30.941, 'epoch': 1.0}
```

&nbsp;

## 少样本分类

通过为每个类别精心标注少量高质量的数据来完成模型训练。
SetFit框架：基于sentence-transformer架构构建，能在训练中动态优化文本表示质量。
步骤：

1. 采集训练数据：训练数据为包含正例和负例的句子对。

2. 微调嵌入模型

3. 训练分类器

```
 # pip install setfit==1.0.3 
 # pip install huggingface_hub==0.23.5
 from setfit import sample_dataset, SetFitModel
from datasets import load_dataset
from setfit import TrainingArguments as SetFitTrainingArguments
from setfit import Trainer as SetFitTrainer
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np


# Load data
tomatoes = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

# We simulate a few-shot setting by sampling 16 examples per class
sampled_train_data = sample_dataset(tomatoes["train"], num_samples=16)

# Load a pre-trained SentenceTransformer model
model_path = 'E:/huggingface/models/all-mpnet-base-v2'
model = SetFitModel.from_pretrained(model_path)

# Define training arguments
args = SetFitTrainingArguments(
    num_epochs=3, # The number of epochs to use for contrastive learning
    num_iterations=20  # The number of text pairs to generate
)
args.eval_strategy = args.evaluation_strategy

# Create trainer
trainer = SetFitTrainer(
    model=model,
    args=args,
    train_dataset=sampled_train_data,
    eval_dataset=test_data,
    metric="f1"
)

# Training loop
trainer.train()

# 评估
y_true = test_data["label"]
predictions = model.predict(test_data["text"])

# 转换为 NumPy 数组
if hasattr(predictions, 'numpy'):
    predictions = predictions.numpy()  # PyTorch Tensor -> NumPy
elif hasattr(predictions, 'detach'):
    predictions = predictions.detach().numpy()

# 处理预测结果
if len(predictions.shape) > 1 and predictions.shape[1] > 1:
    y_pred = np.argmax(predictions, axis=1)
else:
    y_pred = np.round(predictions).astype(int)

# 计算指标
print("\n" + "=" * 60)
print("评估结果：")
print("=" * 60)
print(classification_report(y_true, y_pred, target_names=["负面", "正面"]))
print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
print(f"F1 (weighted): {f1_score(y_true, y_pred, average='weighted'):.4f}")
print("=" * 60)
```

```
***** Running training *****
  Num unique pairs = 1280
  Batch size = 16
  Num epochs = 3
  Total optimization steps = 240
  0%|          | 0/240 [00:00<?, ?it/s]
。。。。。                                               
100%|██████████| 240/240 [00:48<00:00,  5.00it/s]
{'train_runtime': 48.0095, 'train_samples_per_second': 79.984, 'train_steps_per_second': 4.999, 'epoch': 3.0}
============================================================
评估结果：
============================================================
              precision    recall  f1-score   support

          负面       0.87      0.80      0.83       533
          正面       0.82      0.88      0.84       533

    accuracy                           0.84      1066
   macro avg       0.84      0.84      0.84      1066
weighted avg       0.84      0.84      0.84      1066

准确率: 0.8386
F1 (weighted): 0.8384
============================================================
```

&nbsp;

## 基于掩码语言建模的继续预训练

在预训练好的BERT模型基础上，继续使用特定领域的数据实施掩码语言建模（MLM）训练。

预训练 -》 在特定领域数据上继续预训练 -》 微调

```
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import pipeline, TrainingArguments, Trainer
from datasets import load_dataset


# Load data
tomatoes = load_dataset("E:/huggingface/datasets/rotten_tomatoes")
train_data, test_data = tomatoes["train"], tomatoes["test"]

# Load a pre-trained SentenceTransformer model
model_path = 'E:/huggingface/models/bert-base-uncased'
# Load model for Masked Language Modeling (MLM)
model = AutoModelForMaskedLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_function(examples):
   return tokenizer(examples["text"], truncation=True)

# Tokenize data
tokenized_train = train_data.map(preprocess_function, batched=True)
tokenized_train = tokenized_train.remove_columns("label")
tokenized_test = test_data.map(preprocess_function, batched=True)
tokenized_test = tokenized_test.remove_columns("label")

# Masking Tokens
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# Training arguments for parameter tuning
training_args = TrainingArguments(
   "model",
   learning_rate=2e-5,
   per_device_train_batch_size=16,
   per_device_eval_batch_size=16,
   num_train_epochs=10,
   weight_decay=0.01,
   save_strategy="epoch",
   report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Save pre-trained tokenizer
tokenizer.save_pretrained("mlm")

# Train model
trainer.train()

# Save updated model
model.save_pretrained("mlm")

# Load and create predictions
mask_filler = pipeline("fill-mask", model=model_path)
preds = mask_filler("What a horrible [MASK]!")

# Print results
for pred in preds:
    print(f">>> {pred['sequence']}")

# Load and create predictions
mask_filler = pipeline("fill-mask", model="mlm")
preds = mask_filler("What a horrible [MASK]!")

# Print results
for pred in preds:
    print(f">>> {pred['sequence']}")
```
