# 提示工程

## 使用文本生成模型

模型：Phi-3-mini

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Load model and tokenizer
model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=500,
    do_sample=False,
)

# Prompt
messages = [
    {"role": "user", "content": "Create a funny joke about chickens."}
]

# Generate the output
output = pipe(messages)
print(output[0]["generated_text"])

# Apply prompt template
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)

# Using a high temperature
output = pipe(messages, do_sample=True, temperature=1)
print(output[0]["generated_text"])

# Using a high top_p
output = pipe(messages, do_sample=True, top_p=1)
print(output[0]["generated_text"])
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:09<00:00,  4.77s/it]
 Why did the chicken join the band? Because it had the drumsticks!
<|user|>
Create a funny joke about chickens.<|end|>
<|endoftext|>
 Why did the chicken join a band?

Because it had the drumsticks, and it wanted to be the poultry stage act!
 Why do chickens like to play cards? Because they've got a lot of "hearts."
```

控制模型输出：

1. do_sample：为False时以确保输出的一致性，只会选择最可能的下一个词元；需要用temperature和top_p时需要设置为True。

2. temperature：决定生成文本的创造性。
   a. 为0时每次都会生成相同的响应，总是选择可能性最大的词；
   b. 较高的值会允许生成可能性更小的词。

3. top_p：核采样，控制LLM可以考虑哪些词元子集的采样技术。会考虑概率最高的若干词元，直到达到其累积概率限制。
   a. 例如设置为0.1，则会从概率最高的词元开始考虑，直到词元的累积概率到达0.1。若设置为1，则会考虑所有词元。
   b. 降低top_p值会增加确定性，增加top_p值则会提高创造性。

4. top_k：控制LLM可以考虑的词元数量。
   a. 若设置为100，则LLM将只考虑可能性最大的100个词元。

|      |             |       |                           |
| ---- | ----------- | ----- | ------------------------- |
| 场景   | temperature | top_p | 描述                        |
| 头脑风暴 | 高           | 高     | 高随机性输出，生成的结果高度多样化         |
| 写邮件  | 低           | 低     | 高确定性输出，可预测、确定的输出          |
| 创意写作 | 高           | 低     | 高随机性输出，但保持连贯性             |
| 翻译   | 低           | 高     | 高确定性输出，具有广泛的词汇范围从而更具语言多样性 |

```
# temperature
"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.

    Examples:

    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    >>> set_seed(0)  # for reproducibility

    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> model.config.pad_token_id = model.config.eos_token_id
    >>> inputs = tokenizer(["Hugging Face Company is"], return_tensors="pt")

    >>> # With temperature=1.0, the default, we consistently get random outputs due to random sampling.
    >>> generate_kwargs = {"max_new_tokens": 10, "do_sample": True, "temperature": 1.0, "num_return_sequences": 2}
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is one of these companies that is going to take a',
    "Hugging Face Company is a brand created by Brian A. O'Neil"]

    >>> # However, with temperature close to 0, it approximates greedy decoding strategies (invariant)
    >>> generate_kwargs["temperature"] = 0.0001
    >>> outputs = model.generate(**inputs, **generate_kwargs)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    ['Hugging Face Company is a company that has been around for over 20 years',
    'Hugging Face Company is a company that has been around for over 20 years']
    ```
    """
```

&nbsp;

## 提示工程

提示词的基本要素：指令、数据、输出指示器、排除标准
提示词优化：

1. 具体性：准确描述想要达到的目标

2. 幻觉处理：可以要求LLM只有在明确知道答案时才生成答案，不知道时给定兜底输出。

3. 顺序：在提示词的开头或结尾放置指令。  比较长的提示词，中间部分容易被遗忘，LLM往往会关注开头部分（首位效应）或结尾部分（近因效应）

更复杂提示词的常见组件：

1. 角色定位：描述LLM扮演什么角色，例如XXX专家

2. 指令：具体的任务

3. 上下文：任务背景和附加信息

4. 输出格式：输出文本的格式

5. 受众：生成文本的目标对象

6. 语气：生成文本中应使用的语气

7. 数据：任务相关的主要数据
   
   &nbsp; 

## 上下文学习：提供示例

为LLM提供一个或多个具体示例。

```
from Load model and tokenizer


# Load model and tokenizer
model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
 model_path,
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline

pipe = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=500,
 do_sample=False,
)

# Use a single example of using the made-up word in a sentence

one_shot_prompt = [
 {
 "role": "user",
 "content": "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
 },
 {
 "role": "assistant",
 "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."
 },
 {
 "role": "user",
 "content": "To 'screeg' something is to swing a sword at it. An example of a sentence that uses the word screeg is:"
 }
]

# Generate the output

outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00, 2.40s/it]
 In the ancient battle, the knight bravely screeged at the approaching enemy, his sword gleaming in the sunlight.
```

&nbsp;

## 链式提示：分解问题

将一个提示词的输出作为下一个提示词的输入，从而创建一个连续的交互链来解决问题。

让LLM将问题分解为多个子问题，能够在每个单独子问题上投入更多时间，而不是一次性解决整个问题。

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# Load model and tokenizer
model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
 model_path,
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline

pipe = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=500,
 do_sample=False,
)

# Create name and slogan for a product

product_prompt = [
 {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
]
outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]
print(product_description)

# Based on a name and slogan for a product, generate a sales pitch

sales_prompt = [
 {"role": "user", "content": f"Generate a very short sales pitch for the following product: '{product_description}'"}
]
outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]
print(sales_pitch)
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:05<00:00, 2.97s/it]
 Name: ChatSage
Slogan: "Your AI Companion for Smart Conversations"
 Introducing ChatSage, your AI companion for smart conversations. With ChatSage, you'll have a personalized and intelligent assistant at your fingertips, ready to engage in meaningful dialogue, provide helpful information, and enhance your communication experience. Say goodbye to awkward silences and hello to effortless, intelligent conversations with ChatSage.
```

&nbsp;

## 使用生成模型进行推理

### 思维链：先思考，再回答

在提示词中包含一个或多个推理示例，或直接要求模型逐步思考（think step by step）。

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer

model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
 model_path,
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline

pipe = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=500,
 do_sample=False,
)

# Answering with chain-of-thought

cot_prompt = [
 {"role": "user", "content": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"},
 {"role": "assistant", "content": "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11."},
 {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?"}
]

# Generate the output

outputs = pipe(cot_prompt)
print(outputs[0]["generated_text"])

# Zero-shot Chain-of-Thought

zeroshot_cot_prompt = [
 {"role": "user", "content": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step."}
]

# Generate the output

outputs = pipe(zeroshot_cot_prompt)
print(outputs[0]["generated_text"])
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00, 2.23s/it]
 The cafeteria started with 23 apples. They used 20 apples to make lunch, so they had 23 - 20 = 3 apples left. After buying 6 more apples, they now have 3 + 6 = 9 apples. The answer is 9.
 Step 1: Start with the initial number of apples in the cafeteria, which is 23.

Step 2: Subtract the number of apples used to make lunch, which is 20.
23 - 20 = 3 apples remaining.

Step 3: Add the number of apples bought, which is 6.
3 + 6 = 9 apples.

So, the cafeteria now has 9 apples.
```

&nbsp;

### 自洽性 self-consistency：采样输出

用相同的提示词多次调用模型，使用不同的temperature和top_p参数来提高采样的多样性，最后通过投票将占多数的结果作为最终答案。

&nbsp;

### 思维树（Tree of Thought, ToT）：探索中间步骤

要求模型通过模拟多个专家之间的对话来将问题分解成多个部分，在每个部分中探索当前问题的不同解决方案并通过投票选出最佳方案。

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
 model_path,
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline

pipe = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=500,
 do_sample=False,
)

# Zero-shot Chain-of-Thought

zeroshot_tot_prompt = [
 {"role": "user", "content": "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave. The question is 'The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?' Make sure to discuss the results."}
]

# Generate the output

outputs = pipe(zeroshot_tot_prompt)
print(outputs[0]["generated_text"])
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00, 2.41s/it]
 Expert 1:
Step 1: Start with the initial number of apples, which is 23.

Expert 2:
Step 1: Subtract the number of apples used for lunch, which is 20.
Step 2: Add the number of apples bought, which is 6.

Expert 3:
Step 1: Start with the initial number of apples, which is 23.
Step 2: Subtract the number of apples used for lunch, which is 20.
Step 3: Add the number of apples bought, which is 6.

Results:
All three experts arrived at the same answer. The cafeteria has 9 apples left (23 - 20 + 6 = 9).
```

&nbsp;

### 输出验证

验证项：

1. 结构化输出：需要输出特定格式，例如json；

2. 输出的有效性：不能有非预期的输出内容

3. 准确性：符合期望的标准和性能，事实准确，无幻觉；

4. 安全和伦理：不能包含不文明用语，隐私信息，偏见等。

控制输出的方法：

1. 示例：提供多个预期的输出示例

2. 语法：控制词元选择过程

3. 微调：对模型进行微调

&nbsp;

### 提供示例

```
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer

model_path = "E:/huggingface/models/LLM-Research--Phi-3-mini-4k-instruct/snapshots/master"
model = AutoModelForCausalLM.from_pretrained(
 model_path,
 device_map="cuda",
 torch_dtype="auto",
 trust_remote_code=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a pipeline

pipe = pipeline(
 "text-generation",
 model=model,
 tokenizer=tokenizer,
 return_full_text=False,
 max_new_tokens=500,
 do_sample=False,
)

# One-shot learning: Providing an example of the output structure

one_shot_template = """Create a short character profile for an RPG game. Make sure to only use this format:

{
 "description": "A SHORT DESCRIPTION",
 "name": "THE CHARACTER'S NAME",
 "armor": "ONE PIECE OF ARMOR",
 "weapon": "ONE OR MORE WEAPONS"
}
"""
one_shot_prompt = [
 {"role": "user", "content": one_shot_template}
]

# Generate the output

outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])
```

```
Loading checkpoint shards: 100%|██████████| 2/2 [00:04<00:00, 2.43s/it]
 {
 "description": "A cunning rogue with a mysterious past, skilled in stealth and deception.",
 "name": "Shadowcloak",
 "armor": "Leather Hood",
 "weapon": "Dagger"
}
```

&nbsp;

### 语法：约束采样

使用 llama-cpp-python 加载语言模型，将 response_format 指定为 json 对象，在底层，它会应用json语法来确保输出符合该格式。

下载模型：[microsoft-Phi-3-mini-4k-instruct-GGUF-smashed](https://modelscope.cn/models/PrunaAI/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed)

```
# pip install llama-cpp-python

import json
from llama_cpp.llama import Llama

model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"

# Load Phi-3

llm = Llama(
 model_path=model_path,
 n_gpu_layers=-1,
 n_ctx=2048,
 verbose=False
)

# Generate output

output = llm.create_chat_completion(
 messages=[
 {"role": "user", "content": "Create a warrior for an RPG in JSON format."},
 ],
 response_format={"type": "json_object"},
 temperature=0,
)['choices'][0]['message']["content"]

# Format as json

json_output = json.dumps(json.loads(output), indent=4)
print(json_output)
```
