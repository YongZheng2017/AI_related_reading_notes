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

```
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import BitsAndBytesConfig, AutoTokenizer, pipeline
from trl import DPOConfig, DPOTrainer


def format_prompt(example):
    """Format the prompt to using the <|user|> template TinyLLama is using"""

    # Format answers
    system = "<|system|>\n" + example['system'] + "</s>\n"
    prompt = "<|user|>\n" + example['input'] + "</s>\n<|assistant|>\n"
    chosen = example['chosen'] + "</s>\n"
    rejected = example['rejected'] + "</s>\n"

    return {
        "prompt": system + prompt,
        "chosen": chosen,
        "rejected": rejected,
    }

# Apply formatting to the dataset and select relatively short answers
dataset_path = "E:/huggingface/datasets/distilabel-intel-orca-dpo-pairs/data/*.parquet"
dpo_dataset = (
    load_dataset("parquet", data_files=dataset_path)
)
dpo_dataset = dpo_dataset["train"]

dpo_dataset = dpo_dataset.filter(
    lambda r:
        r["status"] != "tie" and
        r["chosen_score"] >= 8 and
        not r["in_gsm8k_train"]
)
dpo_dataset = dpo_dataset.map(format_prompt, remove_columns=dpo_dataset.column_names)

# 4-bit quantization configuration - Q in QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit precision model loading
    bnb_4bit_quant_type="nf4",  # Quantization type
    bnb_4bit_compute_dtype="float16",  # Compute dtype
    bnb_4bit_use_double_quant=True,  # Apply nested quantization
)

# Merge LoRA and base model
pretrained_model_path = "F:/Practice/Python_Practice/pytorch_practice/TinyLlama-1.1B-qlora"
model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_path,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=bnb_config,
)
merged_model = model.merge_and_unload()

# Load LLaMA tokenizer
model_name = "E:/huggingface/models/TinyLlama-1.1B-intermediate-step-1431k-3T"
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
training_arguments = DPOConfig(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    warmup_ratio=0.1
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_arguments,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=512,
    max_length=512,
)

# Fine-tune model with DPO
dpo_trainer.train()

# Save adapter
dpo_trainer.model.save_pretrained("TinyLlama-1.1B-dpo-qlora")

# Merge LoRA and base model
model = AutoPeftModelForCausalLM.from_pretrained(
    pretrained_model_path,
    low_cpu_mem_usage=True,
    device_map="auto",
)
sft_model = model.merge_and_unload()

# Merge DPO LoRA and SFT model
dpo_model = PeftModel.from_pretrained(
    sft_model,
    "TinyLlama-1.1B-dpo-qlora",
    device_map="auto",
)
dpo_model = dpo_model.merge_and_unload()

# Use our predefined prompt template
prompt = """<|user|>
Tell me something about Large Language Models.</s>
<|assistant|>
"""

# Run our instruction-tuned model
pipe = pipeline(task="text-generation", model=dpo_model, tokenizer=tokenizer)
print(pipe(prompt)[0]["generated_text"])
```

```
  5%|▌         | 10/200 [00:53<16:18,  5.15s/it]{'loss': 0.6924, 'grad_norm': 1.7870515584945679, 'learning_rate': 4.5e-06, 'rewards/chosen': 0.0007463550427928567, 'rewards/rejected': -0.000778942194301635, 'rewards/accuracies': 0.21250000596046448, 'rewards/margins': 0.0015252971788868308, 'logps/rejected': -113.88482666015625, 'logps/chosen': -106.79069519042969, 'logits/rejected': -2.915705680847168, 'logits/chosen': -2.849595069885254, 'epoch': 0.01}
 10%|█         | 20/200 [01:47<16:23,  5.47s/it]{'loss': 0.6771, 'grad_norm': 2.3161356449127197, 'learning_rate': 9.5e-06, 'rewards/chosen': 0.008611917495727539, 'rewards/rejected': -0.024700652807950974, 'rewards/accuracies': 0.44999998807907104, 'rewards/margins': 0.03331257030367851, 'logps/rejected': -158.19476318359375, 'logps/chosen': -125.73799896240234, 'logits/rejected': -3.0606935024261475, 'logits/chosen': -2.922976493835449, 'epoch': 0.03}
 15%|█▌        | 30/200 [02:43<16:02,  5.66s/it]{'loss': 0.6471, 'grad_norm': 2.418865919113159, 'learning_rate': 9.938441702975689e-06, 'rewards/chosen': 0.01567707397043705, 'rewards/rejected': -0.08606791496276855, 'rewards/accuracies': 0.44999998807907104, 'rewards/margins': 0.10174499452114105, 'logps/rejected': -144.3031768798828, 'logps/chosen': -107.13211822509766, 'logits/rejected': -2.9469082355499268, 'logits/chosen': -2.837630271911621, 'epoch': 0.04}
 20%|██        | 40/200 [03:31<13:20,  5.01s/it]{'loss': 0.6081, 'grad_norm': 2.0197136402130127, 'learning_rate': 9.755282581475769e-06, 'rewards/chosen': 0.0045336573384702206, 'rewards/rejected': -0.202768474817276, 'rewards/accuracies': 0.5375000238418579, 'rewards/margins': 0.20730213820934296, 'logps/rejected': -166.26864624023438, 'logps/chosen': -129.5735321044922, 'logits/rejected': -2.9638590812683105, 'logits/chosen': -2.85215425491333, 'epoch': 0.05}
 25%|██▌       | 50/200 [04:26<13:15,  5.30s/it]{'loss': 0.6004, 'grad_norm': 3.369025945663452, 'learning_rate': 9.414737964294636e-06, 'rewards/chosen': -0.046127550303936005, 'rewards/rejected': -0.29416295886039734, 'rewards/accuracies': 0.5, 'rewards/margins': 0.24803538620471954, 'logps/rejected': -168.34597778320312, 'logps/chosen': -134.97463989257812, 'logits/rejected': -2.9724717140197754, 'logits/chosen': -2.8478710651397705, 'epoch': 0.07}
 30%|███       | 60/200 [05:22<13:31,  5.80s/it]{'loss': 0.6148, 'grad_norm': 4.472890377044678, 'learning_rate': 8.94005376803361e-06, 'rewards/chosen': -0.07834647595882416, 'rewards/rejected': -0.3229285180568695, 'rewards/accuracies': 0.36250001192092896, 'rewards/margins': 0.24458202719688416, 'logps/rejected': -125.59132385253906, 'logps/chosen': -108.86293029785156, 'logits/rejected': -2.9801559448242188, 'logits/chosen': -2.9046967029571533, 'epoch': 0.08}
 35%|███▌      | 70/200 [06:18<12:24,  5.72s/it]{'loss': 0.5937, 'grad_norm': 1.313569188117981, 'learning_rate': 8.345653031794292e-06, 'rewards/chosen': -0.0611492395401001, 'rewards/rejected': -0.4165809750556946, 'rewards/accuracies': 0.375, 'rewards/margins': 0.3554316759109497, 'logps/rejected': -136.60519409179688, 'logps/chosen': -108.34822845458984, 'logits/rejected': -3.0213749408721924, 'logits/chosen': -2.937819719314575, 'epoch': 0.09}
 40%|████      | 80/200 [07:15<11:50,  5.92s/it]{'loss': 0.534, 'grad_norm': 1.409082055091858, 'learning_rate': 7.649596321166024e-06, 'rewards/chosen': -0.0853465124964714, 'rewards/rejected': -0.6397562623023987, 'rewards/accuracies': 0.5, 'rewards/margins': 0.5544098019599915, 'logps/rejected': -168.27525329589844, 'logps/chosen': -135.24526977539062, 'logits/rejected': -2.928148031234741, 'logits/chosen': -2.7913906574249268, 'epoch': 0.11}
 45%|████▌     | 90/200 [08:10<10:02,  5.47s/it]{'loss': 0.5597, 'grad_norm': 3.73032283782959, 'learning_rate': 6.873032967079562e-06, 'rewards/chosen': -0.11632146686315536, 'rewards/rejected': -0.6740682125091553, 'rewards/accuracies': 0.4375, 'rewards/margins': 0.5577467679977417, 'logps/rejected': -175.5201873779297, 'logps/chosen': -116.31217956542969, 'logits/rejected': -3.0396411418914795, 'logits/chosen': -2.861473560333252, 'epoch': 0.12}
 50%|█████     | 100/200 [09:07<09:51,  5.92s/it]{'loss': 0.6354, 'grad_norm': 1.0067471265792847, 'learning_rate': 6.039558454088796e-06, 'rewards/chosen': -0.21031954884529114, 'rewards/rejected': -0.7997227907180786, 'rewards/accuracies': 0.4625000059604645, 'rewards/margins': 0.5894031524658203, 'logps/rejected': -183.30694580078125, 'logps/chosen': -129.44375610351562, 'logits/rejected': -2.932190418243408, 'logits/chosen': -2.7918035984039307, 'epoch': 0.14}
 55%|█████▌    | 110/200 [10:05<08:40,  5.78s/it]{'loss': 0.4996, 'grad_norm': 1.5949487686157227, 'learning_rate': 5.174497483512506e-06, 'rewards/chosen': -0.07178173959255219, 'rewards/rejected': -0.8971213102340698, 'rewards/accuracies': 0.574999988079071, 'rewards/margins': 0.8253396153450012, 'logps/rejected': -198.72897338867188, 'logps/chosen': -146.10479736328125, 'logits/rejected': -2.943246364593506, 'logits/chosen': -2.779600143432617, 'epoch': 0.15}
 60%|██████    | 120/200 [11:04<07:50,  5.88s/it]{'loss': 0.5859, 'grad_norm': 5.5328521728515625, 'learning_rate': 4.304134495199675e-06, 'rewards/chosen': -0.17969359457492828, 'rewards/rejected': -0.9202000498771667, 'rewards/accuracies': 0.48750001192092896, 'rewards/margins': 0.7405065298080444, 'logps/rejected': -194.0130615234375, 'logps/chosen': -123.16999816894531, 'logits/rejected': -2.859191417694092, 'logits/chosen': -2.6360604763031006, 'epoch': 0.16}
 65%|██████▌   | 130/200 [11:56<06:07,  5.25s/it]{'loss': 0.6298, 'grad_norm': 2.4332029819488525, 'learning_rate': 3.4549150281252635e-06, 'rewards/chosen': -0.1871483027935028, 'rewards/rejected': -0.6710089445114136, 'rewards/accuracies': 0.4124999940395355, 'rewards/margins': 0.483860582113266, 'logps/rejected': -146.4880828857422, 'logps/chosen': -122.91691589355469, 'logits/rejected': -2.95247745513916, 'logits/chosen': -2.8950753211975098, 'epoch': 0.18}
 70%|███████   | 140/200 [12:50<05:40,  5.68s/it]{'loss': 0.5868, 'grad_norm': 3.679938793182373, 'learning_rate': 2.6526421860705474e-06, 'rewards/chosen': -0.1373380571603775, 'rewards/rejected': -0.7907261848449707, 'rewards/accuracies': 0.4375, 'rewards/margins': 0.6533880829811096, 'logps/rejected': -140.21502685546875, 'logps/chosen': -127.44154357910156, 'logits/rejected': -3.0303025245666504, 'logits/chosen': -2.941232442855835, 'epoch': 0.19}
 75%|███████▌  | 150/200 [13:40<04:18,  5.16s/it]{'loss': 0.5748, 'grad_norm': 0.7100883722305298, 'learning_rate': 1.9216926233717087e-06, 'rewards/chosen': -0.07069531083106995, 'rewards/rejected': -0.7885652780532837, 'rewards/accuracies': 0.4000000059604645, 'rewards/margins': 0.7178699374198914, 'logps/rejected': -141.05581665039062, 'logps/chosen': -81.0375747680664, 'logits/rejected': -2.905264377593994, 'logits/chosen': -2.7816669940948486, 'epoch': 0.2}
 80%|████████  | 160/200 [14:33<03:38,  5.46s/it]{'loss': 0.5889, 'grad_norm': 2.0510945320129395, 'learning_rate': 1.2842758726130283e-06, 'rewards/chosen': -0.09275250136852264, 'rewards/rejected': -0.7196430563926697, 'rewards/accuracies': 0.42500001192092896, 'rewards/margins': 0.6268905997276306, 'logps/rejected': -148.88931274414062, 'logps/chosen': -107.92057800292969, 'logits/rejected': -2.9309070110321045, 'logits/chosen': -2.822575092315674, 'epoch': 0.22}
 85%|████████▌ | 170/200 [15:27<02:48,  5.62s/it]{'loss': 0.6072, 'grad_norm': 2.145738363265991, 'learning_rate': 7.597595192178702e-07, 'rewards/chosen': -0.06866542994976044, 'rewards/rejected': -0.6307964324951172, 'rewards/accuracies': 0.3499999940395355, 'rewards/margins': 0.5621310472488403, 'logps/rejected': -129.4966583251953, 'logps/chosen': -100.701171875, 'logits/rejected': -2.9245519638061523, 'logits/chosen': -2.8931288719177246, 'epoch': 0.23}
 90%|█████████ | 180/200 [16:19<01:39,  4.96s/it]{'loss': 0.6253, 'grad_norm': 5.127623081207275, 'learning_rate': 3.6408072716606346e-07, 'rewards/chosen': -0.19563503563404083, 'rewards/rejected': -0.7856829166412354, 'rewards/accuracies': 0.4124999940395355, 'rewards/margins': 0.5900478363037109, 'logps/rejected': -152.93312072753906, 'logps/chosen': -132.02328491210938, 'logits/rejected': -2.938133716583252, 'logits/chosen': -2.9063563346862793, 'epoch': 0.24}
 95%|█████████▌| 190/200 [17:09<00:51,  5.11s/it]{'loss': 0.6721, 'grad_norm': 5.507063388824463, 'learning_rate': 1.0926199633097156e-07, 'rewards/chosen': -0.12195022404193878, 'rewards/rejected': -0.4330800473690033, 'rewards/accuracies': 0.3375000059604645, 'rewards/margins': 0.3111298680305481, 'logps/rejected': -123.92408752441406, 'logps/chosen': -101.3538818359375, 'logits/rejected': -2.823333978652954, 'logits/chosen': -2.779634475708008, 'epoch': 0.26}
100%|██████████| 200/200 [18:02<00:00,  5.41s/it]
{'loss': 0.5555, 'grad_norm': 2.839907169342041, 'learning_rate': 3.0458649045211897e-09, 'rewards/chosen': -0.021109113469719887, 'rewards/rejected': -0.7731715440750122, 'rewards/accuracies': 0.4749999940395355, 'rewards/margins': 0.7520624399185181, 'logps/rejected': -161.9045867919922, 'logps/chosen': -107.6459732055664, 'logits/rejected': -2.965052604675293, 'logits/chosen': -2.816384792327881, 'epoch': 0.27}
{'train_runtime': 1082.5214, 'train_samples_per_second': 1.478, 'train_steps_per_second': 0.185, 'train_loss': 0.6044228625297546, 'epoch': 0.27}
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
