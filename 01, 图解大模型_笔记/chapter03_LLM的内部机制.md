# LLM的内部机制

## Transformer模型概述

- 接收文本输入，生成文本输出；  
- 一次只生成一个词元；
- 每个词元生成都是模型的一次前行传播（输入文本进入神经网络并流经计算图，最终产生输出的计算过程）
- 生成词元后，将输出词元追加到输入提示词的末尾，从而调整下一次生成的输入提示词
- 自回归模型：使用早期预测来进行后续预测的模型
  
  &nbsp;

## 前向传播的组成

- 分词器
  - 包含一个词元表；
  - 为每个词元表中的词元关联一个向量表示（词元嵌入）；
- 堆叠的Transformer块
- 语言建模头
  - 一个简单的神经网络层，将Transformer块的输出转换为预测下一个词元的概率分布；
  - 有多种类型，例如序列分类头、词元分类头，用于构建不同类型的系统。

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

model_path = Path("E:/modelscope/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
    local_files_only=True,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

print(model)
```

```
Phi3ForCausalLM(
  (model): Phi3Model(
    (embed_tokens): Embedding(32064, 3072, padding_idx=32000)
    (layers): ModuleList(
      (0-31): 32 x Phi3DecoderLayer(
        (self_attn): Phi3Attention(
          (o_proj): Linear(in_features=3072, out_features=3072, bias=False)
          (qkv_proj): Linear(in_features=3072, out_features=9216, bias=False)
        )
        (mlp): Phi3MLP(
          (gate_up_proj): Linear(in_features=3072, out_features=16384, bias=False)
          (down_proj): Linear(in_features=8192, out_features=3072, bias=False)
          (activation_fn): SiLUActivation()
        )
        (input_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (post_attention_layernorm): Phi3RMSNorm((3072,), eps=1e-05)
        (resid_attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_mlp_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (norm): Phi3RMSNorm((3072,), eps=1e-05)
    (rotary_emb): Phi3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3072, out_features=32064, bias=False)
)
```

词元嵌入：(embed_tokens): Embedding(32064, 3072, padding_idx=32000)，有32064个词元，每个词元的向量大小为3072；  

堆叠的Transformer解码器层：32 x Phi3DecoderLayer，32个Phi3DecoderLayer类型的块；每个块都包含一个注意力层（Phi3Attention）和一个前馈神经网络层（Phi3MLP）  

语言建模头 lm_head：Linear(in_features=3072, out_features=32064, bias=False)，接收一个大小为3072的向量，输出大小等于词表中词元数量（32064）的向量，该输出是每个词元的概率分数。

&nbsp;

## 从概率分布中选择单个词元

解码策略：从概率分布中选择单个词元的方法。

贪心解码：每次都选择概率分数最高的词元（temperature设置为0时）

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path

model_path = Path("E:/modelscope/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
    local_files_only=True,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

prompt = "The capital of France is"

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# 将词元移动到GPU上
input_ids = input_ids.to("cuda")

# Get the output of the model before the lm_head
model_output = model.model(input_ids)

# Get the output of the lm_head
lm_head_output = model.lm_head(model_output[0])

token_id = lm_head_output[0,-1].argmax(-1)
print(tokenizer.decode(token_id))
```

&nbsp;

## 并行词元处理和上下文长度

Transformer特性之一：更适合并行计算  

模型的上下文长度：一次可以处理的最大词元数量  

每个词元都会流经自己的计算路径（独立的计算流），对于文本生成来说，只有最后一条计算流的输出结果用于预测下一个词元，但Transformer的注意力机制会使用之前的计算流结果。

&nbsp;

## 通过缓存键-值加速生成过程

模型缓存之前的计算结果，特别是注意力机制中的一些特定向量，生成词元时就只需要计算最后一条流了，从而显著加速生成过程。

默认启用了缓存，可通过use_cache参数设置。

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import time

model_path = Path("E:/modelscope/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path,
    local_files_only=True,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

prompt = "Write a very long email apologizing to Sarah for the tragic gardening mishap. Explain how it happened."

# Tokenize the input prompt
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

start_time = time.time()
# Generate the text
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=100,
  use_cache=True
)
first_time = time.time() - start_time

start_time = time.time()
# Generate the text
generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=100,
  use_cache=False
)
second_time = time.time() - start_time

# 打印结果
print(f"使用缓存 (use_cache=True) 耗时: {first_time:.4f} 秒")
print(f"不使用缓存 (use_cache=False) 耗时: {second_time:.4f} 秒")
print(f"性能提升: {((second_time - first_time) / second_time * 100):.2f}%")
```

```
Loading weights: 100%|██████████| 195/195 [00:04<00:00, 46.42it/s]
使用缓存 (use_cache=True) 耗时: 46.1143 秒
不使用缓存 (use_cache=False) 耗时: 63.1314 秒
性能提升: 26.96%
```

&nbsp;

## Transformer块的内部结构

Transformer LLM由一系列Transformer块堆叠组成，原始Transformer论文中约为6个，许多LLM中超过100个。

每个块处理其输入，然后将结果传递给下一个块。

每个块由首尾相接的两个组件构成：

- 自注意力层：主要负责整合来自其他输入词元和位置相关的信息。帮助模型在处理特定词元时整合上下文信息。
  
  - 主要处理步骤：
    
    - 1，对当前处理的词元与之前输入词元的相关性评分；
    
    - 2，利用这些评分，将不同位置的信息组合成单一的输出向量。
  
  - 注意力头（attention head）：注意力机制被复制多份，并行执行，这些并行执行的注意力执行过程称为注意力头。提高了模型对输入序列中复杂模式的建模能力，使其能同时关注不同的模式。
  
  - 计算方式：
    
    - 训练过程会产生三个投影矩阵：查询投影矩阵、键投影矩阵、值投影矩阵；
    
    - 注意力首先将输入与投影矩阵相乘，得到三个新的矩阵，即查询矩阵、键矩阵和值矩阵，这些矩阵包含了投影到三个不同空间的输入词元信息。
  
  - 相关性评分：通过将当前位置的查询向量与键矩阵相乘来实现，产生一组分数用以衡量当前位置之前的每一个词元的相关性，再通过softmax操作进行归一化，使它们的和为1。
  
  - 信息组合：用每个词元对应的值向量乘以该词元的分数，然后将这些结果向量相加。

- 前馈神经网络层：模型的主要处理能力。

&nbsp;

## Transformer架构的最新改进

- 更高效的注意力机制
  
  - 稀疏注意力/滑动窗口注意力：只关注少量前序位置（GPT-3交替使用全局注意力和稀疏注意力）
  
  - 多查询注意力：通过在所有注意力头之间共享键矩阵和值矩阵来实现优化。
  
  - 分组查询注意力：分组共享键矩阵和值矩阵
  
  - Flash Attention：通过优化GPU共享内存和高带宽内存之间的数据加载和迁移来加速注意力计算。

- Transformer块
  
  - 预归一化：通过RMSNorm实现
  
  - SwiGLU代替传统的ReLU激活函数

- 位置嵌入
  
  - RoPE：旋转位置嵌入 rotary position embedding
