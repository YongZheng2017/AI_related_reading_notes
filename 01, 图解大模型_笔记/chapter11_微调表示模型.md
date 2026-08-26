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

```
Map: 100%|██████████| 1066/1066 [00:00<00:00, 15960.79 examples/s]
  9%|▉         | 500/5340 [01:17<13:10,  6.12it/s]{'loss': 2.515, 'grad_norm': 15.747591972351074, 'learning_rate': 1.812734082397004e-05, 'epoch': 0.94}
 10%|▉         | 533/5340 [01:22<11:58,  6.69it/s]Checkpoint destination directory model\checkpoint-534 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 19%|█▊        | 1000/5340 [02:37<12:07,  5.96it/s]{'loss': 2.3949, 'grad_norm': 14.226686477661133, 'learning_rate': 1.6254681647940076e-05, 'epoch': 1.87}
 20%|█▉        | 1067/5340 [02:47<10:54,  6.53it/s]Checkpoint destination directory model\checkpoint-1068 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 28%|██▊       | 1500/5340 [03:56<09:09,  6.99it/s]{'loss': 2.2978, 'grad_norm': 19.81380844116211, 'learning_rate': 1.4382022471910113e-05, 'epoch': 2.81}
 30%|██▉       | 1601/5340 [04:10<09:18,  6.70it/s]Checkpoint destination directory model\checkpoint-1602 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 37%|███▋      | 2000/5340 [05:17<07:56,  7.01it/s]{'loss': 2.174, 'grad_norm': 14.12675952911377, 'learning_rate': 1.250936329588015e-05, 'epoch': 3.75}
 40%|███▉      | 2135/5340 [05:38<08:47,  6.08it/s]Checkpoint destination directory model\checkpoint-2136 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 47%|████▋     | 2500/5340 [06:36<07:13,  6.55it/s]{'loss': 2.114, 'grad_norm': 15.20629596710205, 'learning_rate': 1.0636704119850187e-05, 'epoch': 4.68}
 50%|████▉     | 2669/5340 [07:01<06:18,  7.06it/s]Checkpoint destination directory model\checkpoint-2670 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 56%|█████▌    | 3000/5340 [07:54<05:47,  6.74it/s]{'loss': 2.1074, 'grad_norm': 25.626550674438477, 'learning_rate': 8.764044943820226e-06, 'epoch': 5.62}
 60%|█████▉    | 3203/5340 [08:26<06:03,  5.89it/s]Checkpoint destination directory model\checkpoint-3204 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 66%|██████▌   | 3500/5340 [09:12<04:31,  6.78it/s]{'loss': 2.0467, 'grad_norm': 13.224017143249512, 'learning_rate': 6.891385767790263e-06, 'epoch': 6.55}
 70%|██████▉   | 3737/5340 [09:48<04:38,  5.75it/s]Checkpoint destination directory model\checkpoint-3738 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 75%|███████▍  | 4000/5340 [10:29<03:21,  6.64it/s]{'loss': 2.0412, 'grad_norm': 15.023694038391113, 'learning_rate': 5.0187265917603005e-06, 'epoch': 7.49}
 80%|███████▉  | 4271/5340 [11:10<02:50,  6.25it/s]Checkpoint destination directory model\checkpoint-4272 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 84%|████████▍ | 4500/5340 [11:48<03:31,  3.98it/s]{'loss': 2.0083, 'grad_norm': 19.5888729095459, 'learning_rate': 3.146067415730337e-06, 'epoch': 8.43}
 90%|████████▉ | 4805/5340 [12:34<01:15,  7.10it/s]Checkpoint destination directory model\checkpoint-4806 already exists and is non-empty. Saving will proceed but saved results may be invalid.
 94%|█████████▎| 5000/5340 [13:05<00:50,  6.76it/s]{'loss': 2.0318, 'grad_norm': 22.393526077270508, 'learning_rate': 1.2734082397003748e-06, 'epoch': 9.36}
100%|█████████▉| 5339/5340 [13:57<00:00,  6.69it/s]Checkpoint destination directory model\checkpoint-5340 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'train_runtime': 838.9925, 'train_samples_per_second': 101.67, 'train_steps_per_second': 6.365, 'train_loss': 2.1630037800649577, 'epoch': 10.0}
100%|██████████| 5340/5340 [13:58<00:00,  6.36it/s]
Some weights of the model checkpoint at E:/huggingface/models/bert-base-uncased were not used when initializing BertForMaskedLM: ['bert.pooler.dense.bias', 'bert.pooler.dense.weight', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
>>> what a horrible idea!
>>> what a horrible thing!
>>> what a horrible day!
>>> what a horrible story!
>>> what a horrible dream!
>>> what a horrible movie!
>>> what a horrible film!
>>> what a horrible thing!
>>> what a horrible idea!
>>> what a horrible picture!
```

&nbsp;

## 命名实体识别

命名实体识别（Named Entity Recognition，NER）：从文本数据中获取人名、地名等实体信息。

数据集：conll2003

下载：huggingface-cli download BramVanroy/conll2003 --repo-type dataset --local-dir E:\huggingface\datasets\conll2003

模型：bert-base-cased

下载模型：modelscope download --model google-bert/bert-base-cased --local_dir E:\huggingface\models\bert-base-cased

```
# pip install spacy-to-hf
from transformers import AutoTokenizer, pipeline, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
import seqeval.metrics as seqeval_metrics


# Load data
dataset = load_dataset("E:/huggingface/datasets/conll2003",
                       trust_remote_code=True)
train_data, test_data = dataset["train"], dataset["test"]

label2id = {
    'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4,
    'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8
}
id2label = {index: label for label, index in label2id.items()}

model_path = 'E:/huggingface/models/bert-base-uncased'
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    model_path,
    num_labels=len(id2label),
    id2label=id2label,
    label2id=label2id
)

def align_labels(examples):
    token_ids = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = examples["ner_tags"]

    updated_labels = []
    for index, label in enumerate(labels):

        # Map tokens to their respective word
        word_ids = token_ids.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:

            # The start of a new word
            if word_idx != previous_word_idx:

                previous_word_idx = word_idx
                updated_label = -100 if word_idx is None else label[word_idx]
                label_ids.append(updated_label)

            # Special token is -100
            elif word_idx is None:
                label_ids.append(-100)

            # If the label is B-XXX we change it to I-XXX
            else:
                updated_label = label[word_idx]
                if updated_label % 2 == 1:
                    updated_label += 1
                label_ids.append(updated_label)

        updated_labels.append(label_ids)

    token_ids["labels"] = updated_labels
    return token_ids

tokenized = dataset.map(align_labels, batched=True)


def compute_metrics(eval_pred):
    """使用 seqeval 计算 NER 评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)

    # 将预测和真实标签转换为字符串列表
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        pred_list = []
        label_list = []
        for token_pred, token_label in zip(prediction, label):
            if token_label != -100:  # 忽略特殊token
                pred_list.append(id2label[token_pred])
                label_list.append(id2label[token_label])
        true_predictions.append(pred_list)
        true_labels.append(label_list)

    # 使用 seqeval 计算各项指标
    results = {
        "f1": seqeval_metrics.f1_score(true_labels, true_predictions),
        "precision": seqeval_metrics.precision_score(true_labels, true_predictions),
        "recall": seqeval_metrics.recall_score(true_labels, true_predictions),
        "accuracy": seqeval_metrics.accuracy_score(true_labels, true_predictions),
    }
    return results

# Token-classification Data Collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()

# Evaluate the model on our test data
trainer.evaluate()

# Save our fine-tuned model
trainer.save_model("ner_model")

# Run inference on the fine-tuned model
token_classifier = pipeline(
    "token-classification",
    model="ner_model",
)
result = token_classifier("My name is Maarten.")
print(result)
```

```
Map: 100%|██████████| 3453/3453 [00:00<00:00, 24326.42 examples/s]
 57%|█████▋    | 500/878 [00:59<00:45,  8.22it/s]{'loss': 0.2177, 'grad_norm': 0.8724098205566406, 'learning_rate': 8.610478359908885e-06, 'epoch': 0.57}
100%|█████████▉| 877/878 [01:44<00:00,  8.77it/s]Checkpoint destination directory model\checkpoint-878 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'train_runtime': 105.4015, 'train_samples_per_second': 133.214, 'train_steps_per_second': 8.33, 'train_loss': 0.15769312213385023, 'epoch': 1.0}
100%|██████████| 878/878 [01:45<00:00,  8.33it/s]
100%|██████████| 216/216 [00:06<00:00, 35.88it/s]
[{'entity': 'B-PER', 'score': np.float32(0.9880164), 'index': 4, 'word': 'ma', 'start': 11, 'end': 13}, {'entity': 'I-PER', 'score': np.float32(0.97985613), 'index': 5, 'word': '##arte', 'start': 13, 'end': 17}, {'entity': 'I-PER', 'score': np.float32(0.98603755), 'index': 6, 'word': '##n', 'start': 17, 'end': 18}]
```
