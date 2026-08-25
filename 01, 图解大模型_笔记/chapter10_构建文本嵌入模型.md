# 构建文本嵌入模型

## 嵌入模型

嵌入（embedding）：将文本数据转换成易于处理的向量。
嵌入模型：将文本数据进行嵌入的模型，目标是尽可能准确地将文本数据表示为嵌入向量。

1. 捕捉文本的语义本质。语义相似性

2. 情感倾向

对比学习：向模型输入相似的和不相似的文档对作为示例，模型需要将一个文档与另一个文档进行对比，进而学习它们之间的相似之处与区别。

&nbsp;

## SBERT

sentence-transformer：对比学习的一种框架，解决了原始BERT在创建句子嵌入时的计算开销问题。
以前的方式：交叉编码器架构，结合BERT模型。
        交叉编码器：允许两个句子同时通过Transformer网络进行处理，以预测两个句子的相似度。通过在原始架构上添加分类头来实现。确定是当需要训练N个句子时，需要进行n*(n-1)/2次推理，产生巨大的开销。
新的方式：sentence-transformer使用孪生（siamese）架构。架构中有两个完全相同的BERT模型，它们共享权重和神经网络架构，这两个模型接收句子作为输入，通过对词元嵌入进行池化来生成嵌入向量，然后根据句子嵌入的相似度进行优化，句子对的优化通过损失函数完成。

&nbsp;

## 构建嵌入模型

通用语言理解评估标准（General Language Understanding Evaluation benchmark, GLUE）
多类型自然语言推理语料库：MNLI

&nbsp;

下载模型：modelscope download --model google-bert/bert-base-uncased --local_dir E:\huggingface\models\bert-base-uncased

```
#  pip install sentence-transformers==3.0.0
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


train_dataset = (
    load_dataset(
        "csv",
        data_files="E:/huggingface/datasets/glue/MNLI/train.tsv",
        sep="\t",
        split="train",
        on_bad_lines="skip"
    )
    .select_columns(['sentence1', 'sentence2', 'gold_label'])
    .rename_columns({'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
    .filter(lambda x: x['label'] is not None and x['label'] != '')  # 过滤空值
    .map(lambda x: {'label': {'entailment': 0, 'neutral': 1, 'contradiction': 2}[x['label']]})
    .select(range(50000))
)

# Use a base model
model_path = "E:/huggingface/models/bert-base-uncased"
embedding_model = SentenceTransformer(model_path)

# Define the loss function. In soft-max loss, we will also need to explicitly set the number of labels.
train_loss = losses.SoftmaxLoss(
    model=embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine",
)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="base_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# Train embedding model
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

# Evaluate our trained model
result = evaluator(embedding_model)
print(result)
```

```
Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
  6%|▋         | 100/1563 [01:28<25:45,  1.06s/it]{'loss': 1.0698, 'grad_norm': 2.517430305480957, 'learning_rate': 5e-05, 'epoch': 0.06}
 13%|█▎        | 200/1563 [03:18<26:58,  1.19s/it]{'loss': 0.9393, 'grad_norm': 4.144111156463623, 'learning_rate': 4.6582365003417636e-05, 'epoch': 0.13}
 19%|█▉        | 300/1563 [05:44<31:11,  1.48s/it]{'loss': 0.8919, 'grad_norm': 4.080749988555908, 'learning_rate': 4.316473000683528e-05, 'epoch': 0.19}
 26%|██▌       | 400/1563 [07:49<57:03,  2.94s/it]{'loss': 0.8481, 'grad_norm': 3.4177963733673096, 'learning_rate': 3.9747095010252904e-05, 'epoch': 0.26}
 32%|███▏      | 500/1563 [09:50<19:09,  1.08s/it]{'loss': 0.8549, 'grad_norm': 4.081290245056152, 'learning_rate': 3.632946001367054e-05, 'epoch': 0.32}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:01,  2.42example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:01,  2.35example/s]
Computing widget examples:  60%|██████    | 3/5 [00:01<00:00,  2.31example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  2.38example/s]
Computing widget examples: 100%|██████████| 5/5 [00:02<00:00,  2.40example/s]
 38%|███▊      | 600/1563 [12:17<19:18,  1.20s/it]{'loss': 0.8337, 'grad_norm': 3.7860238552093506, 'learning_rate': 3.2946001367054005e-05, 'epoch': 0.38}
 45%|████▍     | 700/1563 [14:26<17:13,  1.20s/it]{'loss': 0.8183, 'grad_norm': 4.255568981170654, 'learning_rate': 2.9528366370471632e-05, 'epoch': 0.45}
 51%|█████     | 800/1563 [16:39<15:35,  1.23s/it]{'loss': 0.8232, 'grad_norm': 4.544357776641846, 'learning_rate': 2.611073137388927e-05, 'epoch': 0.51}
 58%|█████▊    | 900/1563 [18:39<13:37,  1.23s/it]{'loss': 0.7993, 'grad_norm': 4.280099868774414, 'learning_rate': 2.2693096377306907e-05, 'epoch': 0.58}
 64%|██████▍   | 1000/1563 [20:50<10:10,  1.08s/it]{'loss': 0.8097, 'grad_norm': 3.1991357803344727, 'learning_rate': 1.9275461380724537e-05, 'epoch': 0.64}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:01,  2.95example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:01,  2.79example/s]
Computing widget examples:  60%|██████    | 3/5 [00:01<00:00,  2.67example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  2.68example/s]
Computing widget examples: 100%|██████████| 5/5 [00:01<00:00,  2.64example/s]
 70%|███████   | 1100/1563 [23:03<09:25,  1.22s/it]{'loss': 0.7978, 'grad_norm': 4.040947437286377, 'learning_rate': 1.5857826384142175e-05, 'epoch': 0.7}
 77%|███████▋  | 1200/1563 [25:02<10:01,  1.66s/it]{'loss': 0.7731, 'grad_norm': 4.49001407623291, 'learning_rate': 1.2440191387559808e-05, 'epoch': 0.77}
 83%|████████▎ | 1300/1563 [26:59<04:56,  1.13s/it]{'loss': 0.7661, 'grad_norm': 3.4192943572998047, 'learning_rate': 9.022556390977444e-06, 'epoch': 0.83}
 90%|████████▉ | 1400/1563 [28:51<02:35,  1.05it/s]{'loss': 0.7799, 'grad_norm': 2.9339191913604736, 'learning_rate': 5.604921394395079e-06, 'epoch': 0.9}
 96%|█████████▌| 1500/1563 [31:15<01:35,  1.52s/it]{'loss': 0.7718, 'grad_norm': 4.332223415374756, 'learning_rate': 2.187286397812714e-06, 'epoch': 0.96}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:01,  2.99example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:01,  2.89example/s]
Computing widget examples:  60%|██████    | 3/5 [00:01<00:00,  2.69example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  2.70example/s]
Computing widget examples: 100%|██████████| 5/5 [00:01<00:00,  2.70example/s]
100%|██████████| 1563/1563 [33:01<00:00,  1.27s/it]
{'train_runtime': 1981.6481, 'train_samples_per_second': 25.232, 'train_steps_per_second': 0.789, 'train_loss': 0.8354914415668236, 'epoch': 1.0}
{'pearson_cosine': np.float64(0.3814552714847892), 'spearman_cosine': np.float64(0.4731552086711547), 'pearson_manhattan': np.float64(0.41864277212890066), 'spearman_manhattan': np.float64(0.4608234670757643), 'pearson_euclidean': np.float64(0.4019447512549862), 'spearman_euclidean': np.float64(0.4533126629667702), 'pearson_dot': np.float64(0.36291430939416963), 'spearman_dot': np.float64(0.3795451334768951), 'pearson_max': np.float64(0.41864277212890066), 'spearman_max': np.float64(0.4731552086711547)}
```

&nbsp;

评估

大规模文本嵌入基准（Massive Text Embedding Benchmark, MTEB）：涵盖8个嵌入任务，58个数据集和112种语言。

```
from mteb import MTEB

# Choose evaluation task
evaluation = MTEB(tasks=["Banking77Classification"])

# Calculate results
results = evaluation.run(embedding_model)
```

&nbsp;

损失函数

通常不建议使用softmax，其他损失函数可能更高效。

余弦相似度损失函数：通常用于语义文本相似度任务。

```
train_loss = losses.CosineSimilarityLoss(model=embedding_model)
```

多负例排序损失函数：

```
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)
```

&nbsp;

## 微调嵌入模型

### 监督学习

模型：all-MiniLM-L6-v2

下载模型：modelscope download --model sentence-transformers/all-MiniLM-L6-v2 --local_dir E:\huggingface\models\sentence-transformers_all-MiniLM-L6-v2

```
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments


train_dataset = (
    load_dataset(
        "csv",
        data_files="E:/huggingface/datasets/glue/MNLI/train.tsv",
        sep="\t",
        split="train",
        on_bad_lines="skip"
    )
    .select_columns(['sentence1', 'sentence2', 'gold_label'])
    .rename_columns({'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
    .filter(lambda x: x['label'] is not None and x['label'] != '')  # 过滤空值
    .map(lambda x: {'label': {'entailment': 0, 'neutral': 1, 'contradiction': 2}[x['label']]})
    .select(range(50000))
)

# Use a base model
model_path = "E:/huggingface/models/sentence-transformers_all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_path)

# Create an embedding similarity evaluator for stsb
val_sts = load_dataset('glue', 'stsb', split='validation')
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine"
)

# Define the loss function. In soft-max loss, we will also need to explicitly set the number of labels.
train_loss = losses.SoftmaxLoss(
    model=embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)

# Define the training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="finetuned_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# Train model
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()

# Evaluate our trained model
result = evaluator(embedding_model)
print(result)

# Evaluate the pre-trained model
original_model = SentenceTransformer(model_path)
evaluator(original_model)
print(result)
```

```
F:\Practice\Python_Practice\pytorch_practice\.venv\Lib\site-packages\accelerate\accelerator.py:450: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler(**kwargs)
  6%|▋         | 100/1563 [00:08<01:48, 13.45it/s]{'loss': 1.0907, 'grad_norm': 0.3566739857196808, 'learning_rate': 5e-05, 'epoch': 0.06}
 13%|█▎        | 200/1563 [00:16<01:57, 11.55it/s]{'loss': 1.0512, 'grad_norm': 0.5452160835266113, 'learning_rate': 4.6582365003417636e-05, 'epoch': 0.13}
 19%|█▉        | 300/1563 [00:24<01:38, 12.84it/s]{'loss': 1.0357, 'grad_norm': 0.713979959487915, 'learning_rate': 4.316473000683528e-05, 'epoch': 0.19}
 26%|██▌       | 400/1563 [00:32<01:33, 12.41it/s]{'loss': 1.0266, 'grad_norm': 0.8041049242019653, 'learning_rate': 3.9747095010252904e-05, 'epoch': 0.26}
 32%|███▏      | 500/1563 [00:39<01:22, 12.95it/s]Checkpoint destination directory finetuned_embedding_model\checkpoint-500 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'loss': 1.0183, 'grad_norm': 1.4437330961227417, 'learning_rate': 3.632946001367054e-05, 'epoch': 0.32}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:01,  3.90example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:00,  3.72example/s]
Computing widget examples:  60%|██████    | 3/5 [00:00<00:00,  3.71example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  3.71example/s]
Computing widget examples: 100%|██████████| 5/5 [00:01<00:00,  3.71example/s]
 38%|███▊      | 600/1563 [01:32<01:13, 13.12it/s]{'loss': 1.0239, 'grad_norm': 1.2388720512390137, 'learning_rate': 3.291182501708818e-05, 'epoch': 0.38}
 45%|████▍     | 700/1563 [01:40<01:12, 11.90it/s]{'loss': 1.0091, 'grad_norm': 0.8875569105148315, 'learning_rate': 2.9494190020505813e-05, 'epoch': 0.45}
 51%|█████     | 800/1563 [01:48<01:06, 11.53it/s]{'loss': 1.0119, 'grad_norm': 1.4020835161209106, 'learning_rate': 2.6076555023923443e-05, 'epoch': 0.51}
 58%|█████▊    | 900/1563 [01:56<00:46, 14.36it/s]{'loss': 1.0059, 'grad_norm': 0.9979203343391418, 'learning_rate': 2.2658920027341084e-05, 'epoch': 0.58}
 64%|██████▍   | 1000/1563 [02:03<00:39, 14.21it/s]Checkpoint destination directory finetuned_embedding_model\checkpoint-1000 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'loss': 1.0074, 'grad_norm': 0.9236011505126953, 'learning_rate': 1.9241285030758715e-05, 'epoch': 0.64}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:00,  4.32example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:00,  4.01example/s]
Computing widget examples:  60%|██████    | 3/5 [00:00<00:00,  3.97example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  3.90example/s]
Computing widget examples: 100%|██████████| 5/5 [00:01<00:00,  3.90example/s]
 70%|███████   | 1100/1563 [02:13<00:37, 12.46it/s]{'loss': 1.0053, 'grad_norm': 1.7060134410858154, 'learning_rate': 1.5823650034176352e-05, 'epoch': 0.7}
 77%|███████▋  | 1200/1563 [02:20<00:29, 12.38it/s]{'loss': 0.997, 'grad_norm': 1.6849638223648071, 'learning_rate': 1.2406015037593984e-05, 'epoch': 0.77}
 83%|████████▎ | 1300/1563 [02:28<00:19, 13.44it/s]{'loss': 0.9991, 'grad_norm': 1.1101402044296265, 'learning_rate': 8.988380041011621e-06, 'epoch': 0.83}
 90%|████████▉ | 1400/1563 [02:36<00:13, 12.31it/s]{'loss': 0.9983, 'grad_norm': 0.5304555296897888, 'learning_rate': 5.570745044429255e-06, 'epoch': 0.9}
 96%|█████████▌| 1500/1563 [02:44<00:05, 11.29it/s]Checkpoint destination directory finetuned_embedding_model\checkpoint-1500 already exists and is non-empty. Saving will proceed but saved results may be invalid.
{'loss': 0.9969, 'grad_norm': 0.9836565256118774, 'learning_rate': 2.15311004784689e-06, 'epoch': 0.96}

Computing widget examples:   0%|          | 0/5 [00:00<?, ?example/s]
Computing widget examples:  20%|██        | 1/5 [00:00<00:00,  4.37example/s]
Computing widget examples:  40%|████      | 2/5 [00:00<00:00,  4.17example/s]
Computing widget examples:  60%|██████    | 3/5 [00:00<00:00,  4.00example/s]
Computing widget examples:  80%|████████  | 4/5 [00:01<00:00,  3.85example/s]
Computing widget examples: 100%|██████████| 5/5 [00:01<00:00,  3.88example/s]
100%|██████████| 1563/1563 [02:50<00:00,  9.16it/s]
{'train_runtime': 170.6181, 'train_samples_per_second': 293.052, 'train_steps_per_second': 9.161, 'train_loss': 1.0176259832205257, 'epoch': 1.0}
{'pearson_cosine': np.float64(0.22483633737083888), 'spearman_cosine': np.float64(0.2773323388303221), 'pearson_manhattan': np.float64(0.25305093767137954), 'spearman_manhattan': np.float64(0.2774507446058801), 'pearson_euclidean': np.float64(0.2535477001551248), 'spearman_euclidean': np.float64(0.2773323388303221), 'pearson_dot': np.float64(0.22483634094310628), 'spearman_dot': np.float64(0.27733354014528444), 'pearson_max': np.float64(0.2535477001551248), 'spearman_max': np.float64(0.2774507446058801)}
{'pearson_cosine': np.float64(0.22483633737083888), 'spearman_cosine': np.float64(0.2773323388303221), 'pearson_manhattan': np.float64(0.25305093767137954), 'spearman_manhattan': np.float64(0.2774507446058801), 'pearson_euclidean': np.float64(0.2535477001551248), 'spearman_euclidean': np.float64(0.2773323388303221), 'pearson_dot': np.float64(0.22483634094310628), 'spearman_dot': np.float64(0.27733354014528444), 'pearson_max': np.float64(0.2535477001551248), 'spearman_max': np.float64(0.2774507446058801)}
```

&nbsp;

### 增强型SBERT

增强型SBERT：只有少量标注数据的情况下微调嵌入模型。利用速度较慢但更精准的交叉编码架构（BERT）来增强和标注更大的输入对集合，这些新标注的数据对被随后用于微调双编码器（SBERT）。
步骤：

1. 使用小型标注数据集（黄金数据集）微调交叉编码器（BERT）；

2. 创建新的句子对；

3. 使用微调后的交叉编码器标注新的句子对（白银数据集）；

4. 在扩展数据集（黄金数据集+白银数据集）上训练双编码器（SBERT）。

&nbsp;

## 无监督学习

TSDAE（Transformer-Generative Pseudo-Labeling）：生成式伪标签。通过删除输入句子中一定比例的词来为其添加噪声，受损句子被输入编码器中，在编码器的基础上有一个池化层，将其映射为句子嵌入，然后解码器尝试重建原始句子，但不包含人为添加的噪声。核心概率是句子嵌入越准确，重建的句子就越准确。

领域适配：目标是将现有的嵌入模型更新到一个包含不同于源领域主题的特定文本领域。一种方法是自适应预训练，先使用无监督学习对特定领域的语料库进行预训练，然后使用目标领域训练数据进行微调。
