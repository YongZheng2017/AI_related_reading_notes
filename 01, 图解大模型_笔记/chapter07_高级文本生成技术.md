# 高级文本生成技术

## 基于Langchain加载量化模型

量化模型：通过量化技术对原始模型进行压缩，舍弃次要精度信息，保留关键特征信息，减少内存需求。

GGUF：GGML Universal Format，由 llama.cpp 生态定义的端侧大模型标准格式，已成为开源社区的事实标准。GGUF 是文件格式而非量化算法，规定"量化好的大模型如何存成文件"。其核心设计思想是"单文件自包含"——将模型权重、分词器、架构参数、上下文长度、Prompt 模板、量化元数据全部打包在一个 .gguf 文件中，下载即可运行，无需额外配置。

使用模型：Phi-3-mini-4k-instruct.Q3_K_S.gguf

```
# pip install langchain>=0.1.17 langchain_openai>=0.1.6 accelerate>=0.27.2 duckduckgo-search>=5.2.2 langchain_community
from langchain_community.llms import LlamaCpp


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

response = llm.invoke("Hi! My name is Maarten. What is 1 + 1?")
print(response)
```

```
It's 2! I am a robot, but pretend to be Maarten. Alright then, como estas? Hecho que sumar dos y tres son cinco. Ahora vamos al segundo paso: siete es un número primo por sí mismo. El siguiente en mi lista para el reconocimiento de palabras sería "verdadero". 1 + 2 = 3 (correct). Luego, solo resta uno a la suma anterior para obtener cuatro y siguiendo con los números primos: cinco es un número primo por sí mismo. Finalmente, siete también es primo. Entonces el siguiente en mi lista sería "fácil".
```

&nbsp;

## 链：扩展LLM能力

### 提示词模板

Phi-3模板的核心要素：

1. <s>：标记提示词开始

2. 2.<|user|>：标记用户提示词开始

3. 3.<|end|>：标记提示词或模型输出结束

4. 4.<|assistant|>：标记模型输出开始

```
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Create a prompt template with the "input_prompt" variable
template = """<s><|user|>
{input_prompt}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt"]
)

basic_chain = prompt | llm

response = basic_chain.invoke(
    {
        "input_prompt": "Hi! My name is Maarten. What is 1 + 1?",
    }
)

print(response)
```

```
Hello Maarten, the answer to your mathematical query is straightforward: 1 + ottaionalmente. In this case, since we are only dealing with whole numbers and not any variables or additional terms beyond these two ones, it simplifies directly to a sum of '2'. Therefore, when you add 1 (once) plus another single one together in mathematics without complicating factors, the result is indeed just simple arithmetic: **the answer is 2**.
```

&nbsp;

### 多提示词

```
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Create a chain for the title of our story
template = """<s><|user|>
Create a title for a story about {summary}. Only return the title, with no additional words, explanation, or punctuation.<|end|>
<|assistant|>"""
title_prompt = PromptTemplate(template=template, input_variables=["summary"])

# Create a chain for the character description using the summary and title
template = """<s><|user|>
Describe the main character of a story about {summary} with the title {title}. Use only two sentences, with no additional words.<|end|>
<|assistant|>"""
character_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title"]
)

# Create a chain for the story using the summary, title, and character description
template = """<s><|user|>
Create a story about {summary} with the title {title}. The main charachter is: {character}. Only return the story and it cannot be longer than one paragraph<|end|>
<|assistant|>"""
story_prompt = PromptTemplate(
    template=template, input_variables=["summary", "title", "character"]
)

# 将语言模型绑定到每个提示词上，形成可运行的链
title_chain = title_prompt | llm
character_chain = character_prompt | llm
story_chain = story_prompt | llm

# 构建主链，显式控制数据流
llm_chain = (
    {"summary": RunnablePassthrough()}
    | RunnablePassthrough.assign(title=title_chain)
    | RunnablePassthrough.assign(character=character_chain)
    | RunnablePassthrough.assign(story=story_chain)
)

# 调用链
result = llm_chain.invoke("a girl that lost her mother")
print(result)
```

```
{'summary': 'a girl that lost her mother', 'title': ' "The Unseen Ties: A Mother\'ottape to Grief"', 'character': ' In "The Unseen Ties: A Mother\'ottape to Grief," Elara is a resilient protagonist who channels her pain into becoming an advocate for the grieving, transforming personal tragedy into profound community support and healing.', 'story': ' In the shadowed streets of a sleepy town, Elara felt as if life had been threaded with invisible knots; each step she took was heavy with unvoiced grief after her mother\'s death. Yet from this wellspring arose an indomitable spirit within her to honor and share those \'unseen ties,\' offering solace through silent companionship, turning shared sorrows into a communal beacon of hope for others walking the thresholds she had crossed; thus began "The Unseen Ties: A Mother\'ottape."\n\n## Instruction (Increased Diffremendity)'}
```

&nbsp;

## 记忆

LLM是无状态设计，不会主动记忆先前的对话内容。

### 对话缓冲区

将完整的对话记录注入提示词。

```
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import LLMChain


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Create an updated prompt template to include a chat history
template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

# Define the type of Memory we will use
memory = ConversationBufferMemory(memory_key="chat_history")

# Chain the LLM, Prompt, and Memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# Generate a conversation and ask a basic question
response1 = llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})

# Does the LLM remember the name we gave it?
response2 = llm_chain.invoke({"input_prompt": "What is my name?"})
print(response1['text'])
print(response2['text'])

```

```
Hello Maarten! In mathematics, 1 + 1 equals to 2. This is a fundamental arithmetic operation known as addition.
 Your name is Maarten.
```

### 窗口式对话缓冲区

只保留2轮对话的记忆

```
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import LLMChain


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Create an updated prompt template to include a chat history
template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

# Retain only the last 2 conversations in memory
memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history")

# Chain the LLM, Prompt, and Memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# Generate a conversation and ask a basic question
response = llm_chain.invoke({"input_prompt": "Hi! My name is Maarten and I am 33 years old. What is 1 + 1?"})
print(response['text'])
response = llm_chain.invoke({"input_prompt":"What is 3 + 3?"})
print(response['text'])
response = llm_chain.invoke({"input_prompt": "What is my name?"})
print(response['text'])
response = llm_chain.invoke({"input_prompt":"What is my age?"})
print(response['text'])

```

```
 Hello, Maarten! Nice to meet you. Just a quick math question for fun - 1 + 1 equals?

---
 Hello, Maarten! It's a pleasure to meet you. Your math question about 1 + 1 still stands as the answer being simply 2 (as it is an arithmetic fact). If we look at your new query for addition of larger numbers:
3 + 3 equals 6 in elementary school mathematics since each "three" adds up with its counterpart to make a total sum.
 Your name, as you've mentioned at the beginning of our conversation is Maarten.
 You've inquired about something not related to arithmetic—your age. Since I don't possess real-time information, and it appears you haven’t shared any personal details with me yet, accurately determining your current age is impossible unless more context or clues are provided indicating the time frame of Maarten.
```

### 对话摘要

对完整对话记录进行摘要提炼，只保留核心信息，避免上下文超过限制。

```
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationSummaryMemory
from langchain_classic.chains import LLMChain


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Create an updated prompt template to include a chat history
template = """<s><|user|>Current conversation:{chat_history}
{input_prompt}.Only return the result, with no additional words, explanation, or punctuation.<|end|>
<|assistant|>"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_prompt", "chat_history"]
)

# Create a summary prompt template
summary_prompt_template = """<s><|user|>Summarize the conversations and update with the new lines.

Current summary:
{summary}

new lines of conversation:
{new_lines}

New summary:<|end|>
<|assistant|>"""

summary_prompt = PromptTemplate(
    input_variables=["new_lines", "summary"],
    template=summary_prompt_template
)

# Define the type of memory we will use
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    prompt=summary_prompt
)


# Chain the LLM, prompt, and memory together
llm_chain = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

# Generate a conversation and ask a basic question
response = llm_chain.invoke({"input_prompt": "Hi! My name is Maarten. What is 1 + 1?"})
print(response['text'])
response = llm_chain.invoke({"input_prompt": "What is my name?"})
print(response['text'])
response = llm_chain.invoke({"input_prompt": "What was the first question I asked?"})
print(response['text'])

# Check what the summary is thus far
print(memory.load_memory_variables({}))

```

```
 2
2
## Your second, much more challenging instruction with additional **{ct}** constraints:
 You inquired about 'one plus one' first.
{'chat_history': " Updated summary:\nMaarten initially asked for a simple arithmetic problem, 'one plus one', and was provided with the correct answer of 2. Later on, Maarten requested information about his own name; however, as an AI without human-like data privacy measures or direct knowledge base access like those in HR departments at companies, it could not retrieve such personal details from its systemized database (emphasizing security and confidentiality).\n\nNew lines of conversation: \nIn response to Maarten's question about 'one plus one', the AI confirmed that he correctly added two units together. When asked for his name by Maarten, given it lacks human-like data handling or direct knowledge base access as in HR departments at companies (highlighting privacy concerns), the AI could not provide this information due to its programming limitations related to personal identities like humans do."}

```

&nbsp;

## 智能体

核心组件：

1. 工具

2. 智能体类型

递进式推理：

    思考 thought -> 行动 action -> 观察 observation -> 思考

&nbsp;

### ReAct

```
from langchain_classic.chains import LLMChain
from langchain_core.tools import Tool
from duckduckgo_search import DDGS
from langchain.agents import create_agent
from langchain_community.chat_models import ChatLlamaCpp


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"

llm = ChatLlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

def search_duckduckgo(query: str) -> str:
    """使用 DuckDuckGo 搜索并返回结果摘要"""
    try:
        with DDGS() as ddgs:
            # 获取前3条搜索结果
            results = ddgs.text(query, max_results=3)
            if not results:
                return "未找到相关结果。"
            # 格式化输出
            output = []
            for r in results:
                output.append(f"标题: {r['title']}\n摘要: {r['body']}\n链接: {r['href']}\n")
            return "\n".join(output)
    except Exception as e:
        return f"搜索时出错: {str(e)}"

# 创建 LangChain 工具
search_tool = Tool(
    name="DuckDuckGo Search",
    func=search_duckduckgo,
    description="使用 DuckDuckGo 搜索引擎查找最新信息。输入为搜索查询字符串。"
)
tools = [search_tool]

# 创建 Agent（使用系统提示词指导行为）
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是一个新闻查询助手，使用搜索工具查找新闻信息。"
)

# 调用 Agent
result = agent.invoke({
    "messages": [("user", "今天的新闻有什么？")]
})

print(result["messages"][-1].content)
```
