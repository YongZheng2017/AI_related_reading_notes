# 微调生成模型

## LLM训练步骤

LLM训练步骤：

1. 语言建模（预训练）：在一个或多个大规模文本数据集上对模型进行预训练，生成基座模型。

2. 第一次微调（监督微调 supervised fine-tuning, SFT）：使基座模型学会遵循指令。

3. 第二次微调（偏好调优 preference tuning）：使模型符合人类偏好的预期行为。

&nbsp;

## 监督微调

全量微调：使用较小已标注的数据集，更新模型是所有参数，使其符合目标任务需求。

参数高效微调（parameter-efficient fine-tuning, PEFT）：

1. 适配器：在Transformer内部引入一组额外的模块化组件，通过微调这些组件来提升模型在特定任务的性能，而无需微调模型的所有权重。

2. 低秩适配（low-rank adaptation, LoRA）：创建基座模型的一个小型子集来进行微调，通过用较小的矩阵近似原始LLM中的大矩阵来创建参数子集。

3. 压缩模型以实现高效训练：减少表示数值的位数，降低精度。

&nbsp;

## 使用QLoRA进行指令微调

数据集：ultrachat_200k

下载数据集：modelscope download --dataset HuggingFaceH4/ultrachat_200k --local_dir E:\huggingface\datasets\ultrachat_200k

模型：TinyLlama/TinyLlama-1.1B-Chat-v1.0，TinyLlama-1.1B-intermediate-step-1431k-3T

下载模型：

modelscope download --model AI-ModelScope/TinyLlama-1.1B-Chat-v1.0 --local-dir E:\huggingface\models\TinyLlama-1.1B-Chat-v1.0

modelscope download --model mathcoder/TinyLlama-1.1B-intermediate-step-1431k-3T --local-dir E:\huggingface\models\TinyLlama-1.1B-intermediate-step-1431k-3T

```
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer


# Load a tokenizer to use its chat template
model_path = "E:/huggingface/models/TinyLlama-1.1B-Chat-v1.0"
template_tokenizer = AutoTokenizer.from_pretrained(model_path)

def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""

    # Format answers
    chat = example["messages"]
    prompt = template_tokenizer.apply_chat_template(chat, tokenize=False)

    return {"text": prompt}

# Load and format the data using the template TinyLLama is using
dataset_path = "E:/huggingface/datasets/ultrachat_200k/data/*.parquet"
dataset = (
    load_dataset("parquet", data_files=dataset_path)
)
dataset = dataset["train"]
dataset = dataset.select(range(3000))
dataset = dataset.map(format_prompt)

# Load the model to train on the GPU
model_name = "E:/huggingface/models/TinyLlama-1.1B-intermediate-step-1431k-3T"  # 基础模型ID

# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit precision model loading
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype="float16",  # Compute dtype
    bnb_4bit_use_double_quant=True,  # Apply nested quantization
)

# Load the model to train on the GPU
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",

    # Leave this out for regular SFT
    quantization_config=bnb_config,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
tokenizer.pad_token = "<PAD>"
tokenizer.padding_side = "left"

# Prepare LoRA Configuration
peft_config = LoraConfig(
    lora_alpha=32,  # LoRA Scaling
    lora_dropout=0.1,  # Dropout for LoRA Layers
    r=64,  # Rank
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=  # Layers to target
     ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

output_dir = "./results"

# Training arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    max_seq_length=512,

    # Leave this out for regular SFT
    peft_config=peft_config,
)

# Train model
trainer.train()

# Save QLoRA weights
trainer.model.save_pretrained("TinyLlama-1.1B-qlora")

model = AutoPeftModelForCausalLM.from_pretrained(
    "TinyLlama-1.1B-qlora",
    low_cpu_mem_usage=True,
    device_map="auto",
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Use our predefined prompt template
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

# Run our instruction-tuned model
pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer)
print(pipe(prompt)[0]["generated_text"])

```

```
  0%|          | 0/375 [00:00<?, ?it/s]F:\Practice\Python_Practice\pytorch_practice\.venv\Lib\site-packages\torch\_dynamo\eval_frame.py:745: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.5 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  return fn(*args, **kwargs)
  3%|▎         | 10/375 [00:31<18:19,  3.01s/it]{'loss': 1.7253, 'grad_norm': 0.2710379660129547, 'learning_rate': 0.00019964928592495045, 'epoch': 0.03}
。。。。。。
100%|██████████| 375/375 [19:35<00:00,  3.13s/it]
{'train_runtime': 1175.4751, 'train_samples_per_second': 2.552, 'train_steps_per_second': 0.319, 'train_loss': 1.4303674023946127, 'epoch': 1.0}
<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
Large Language Models (LLMs) are a type of artificial intelligence (AI) that can generate human-like language. They are trained on large amounts of text data, and they can be used to generate text in various contexts, such as chatbots, machine translation, and natural language processing (NLP).

LLMs are based on neural networks, which are a type of artificial neural network (ANN) that can simulate the way the human brain works. They are trained on large amounts of text data, and they are able to learn patterns and relationships between words, sentences, and paragraphs.

One of the most important features of LLMs is their ability to generate human-like language. They can generate text that is grammatically correct, has the right tone, and is understandable by humans. This is because LLMs are trained on large amounts of text data, which includes a wide range of language styles and contexts.

Another important feature of LLMs is their ability to generate text in different languages. They can be trained on text in multiple languages, and they can be used to generate text in different contexts, such as chatbots, machine translation, and natural language processing.

LLMs are also capable of generating text in different styles and genres, such as news articles, blog posts, and social media posts. They can be used to generate text in a variety of contexts, such as marketing campaigns, blog posts, and social media posts.

Overall, LLMs are a powerful tool for generating human-like language, and they are being used in a wide range of applications, including chatbots, machine translation, and natural language processing. They are also being used to generate text in different contexts, such as news articles, blog posts, and social media posts.

```

&nbsp;

## 评估生成模型

词级指标

1. 困惑度（perplexity）

2. ROUGE

3. BLEU

4. BERTScore

基准测试

1. MMLU：57个不同任务，包括分类，问答和情感分析

2. GLUE：涵盖各种难度的语言理解任务

3. GSM8K：小学数学应用题

4. HellaSwag：评估常识推理能力

5. HumanEval：164个编程问题

6. TruthfulQA：衡量模型生成文本的真实性

排行榜

1. Open LLM Leaderboard

自动评估：LLM-as-a-judge，让另一个LLM来评判待评估LLM的质量。
人工评估

1. Chatbot Arena：众包投票

&nbsp;

## 偏好调优

通过训练奖励模型（reward model）来实现。

给奖励模型输入一个提示词和一个生成内容，模型输出一个单一的数值，表示该生成内容对于该提示词的偏好/质量。

训练奖励模型：

1. 构建偏好训练数据集

2. 对被接受的生成内容评分

3. 对被拒绝的生成内容评分

使用训练好的奖励模型来微调LLM的一种常用方法是近端策略优化（proximal policy optimization, PPO），一种流行的强化学习技术，通过确保LLM不会过度偏离预期奖励来优化经过指令微调的LLM。

直接偏好优化（direct preference optimization, DPO）是PPO的一种替代方案，摒弃了基于强化学习的训练过程，不需要另外训练奖励模型，而是使用LLM的一个副本作为参考模型，评判参考模型和可训练模型在被接受的生成内容和被拒绝的生成内容的质量方面的偏移。与PPO相比，DPO在训练过程中更加稳定，准确度也更高。

&nbsp;

## 使用DPO进行偏好调优

数据集：distilabel-intel-orca-dpo-pairs

下载数据集：modelscope download --dataset AI-ModelScope/distilabel-intel-orca-dpo-pairs --local_dir E:\huggingface\datasets\distilabel-intel-orca-dpo-pairs
