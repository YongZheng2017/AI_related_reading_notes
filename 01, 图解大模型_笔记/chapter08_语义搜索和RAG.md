# 语义搜索和RAG

语义搜索：通过语义理解而非关键词匹配来实现精准搜索。

语言模型在搜索领域的应用：

1. 稠密搜索（dense retrieval）：基于文本嵌入，将搜索问题转化为查询向量与文档向量的最近邻匹配问题。

2. 重排序（reranking）：接收搜索查询与初始结果集，并根据相关性进行重新排序。

3. RAG：通过整合搜索机制来增强文本生成功能，可有效抑制幻觉，并提升事实准确性。

&nbsp;

## 稠密搜索

过程：

1. 获取文本语料库并进行分块处理

2. 嵌入文本片段

3. 构建搜索索引

4. 通过索引进行搜索

&nbsp;

FAISS：由 Meta（原 Facebook）FAIR 实验室于 2015 年开源的‌高维向量相似性搜索与聚类算法库‌（非完整数据库），核心解决十亿级向量毫秒级检索问题。

https://github.com/facebookresearch/faiss

```
import faiss

dim = embeds.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.float32(embeds))

# Retrieve the nearest neighbors
distances , similar_item_ids = index.search(np.float32([query_embed]), number_of_results)
```

&nbsp;

稠密检索的缺陷：

1. 文本中完全不存在答案时仍会返回结果。需设置相关性阈值。

2. 无法精准匹配特定短语。建议采用混合搜索（结合语义搜索和关键词搜索）。

3. 应用于训练数据之外的领域时，性能会显著下降。



长文本分块策略：

1. 单文档单向量：使用单个向量表示整个文档。
   a. 仅嵌入文档的代表性段落，忽略剩余内容。仅用于快速搭建演示系统。
   b. 将文档分割成多个块并对各个块进行嵌入处理，随后将这些块聚合为单个向量。常用聚合方式是对各向量取平均值，但会高度压缩信息，丢失大量细节。

2. 单文档多向量：将文档切分为较小的块并进行块级嵌入。
   a. 分块方式：
   
     i. 每个句子一块：可能粒度过小，导致向量无法捕获足够的上下文信息。
     ii. 每个段落一块：段落较短时效果较好；段落较长时建议每3~8个句子分为一块。
   b.增强上下文相关性
     i.在块中附加文档标题
     ii.通过构建重叠块结构（相邻块包含部分重叠文本），引入一部分上下文内容。

&nbsp;

专为向量检索设计的数据库：

1. https://github.com/weaviate/weaviate 

2. [ https://www.pinecone.io/](https://www.pinecone.io/)

&nbsp;

## 重排序

```
query = "how precise was the science"
results = co.rerank(query=query, documents=texts, top_n=3, return_documents=True)
```

&nbsp;

## RAG

《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》https://arxiv.org/pdf/2005.11401

将用户的问题与检索获得的前若干相关文档共同输入LLM，使其基于检索提供的上下文生成答案。

&nbsp;

嵌入模型使用：BAAI/bge-small-en-v1.5

下载模型：modelscope download --model BAAI/bge-small-en-v1.5 --local-dir E:\huggingface\models\BAAI_bge-small-en-v1.5

```
# pip install langchain-huggingface[full]
# pip install faiss-cpu
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA


text = """
Interstellar is a 2014 epic science fiction film co-written, directed, and produced by Christopher Nolan.
It stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, Matt Damon, and Michael Caine.
Set in a dystopian future where humanity is struggling to survive, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

Brothers Christopher and Jonathan Nolan wrote the screenplay, which had its origins in a script Jonathan developed in 2007.
Caltech theoretical physicist and 2017 Nobel laureate in Physics[4] Kip Thorne was an executive producer, acted as a scientific consultant, and wrote a tie-in book, The Science of Interstellar.
Cinematographer Hoyte van Hoytema shot it on 35 mm movie film in the Panavision anamorphic format and IMAX 70 mm.
Principal photography began in late 2013 and took place in Alberta, Iceland, and Los Angeles.
Interstellar uses extensive practical and miniature effects and the company Double Negative created additional digital effects.

Interstellar premiered on October 26, 2014, in Los Angeles.
In the United States, it was first released on film stock, expanding to venues using digital projectors.
The film had a worldwide gross over $677 million (and $773 million with subsequent re-releases), making it the tenth-highest grossing film of 2014.
It received acclaim for its performances, direction, screenplay, musical score, visual effects, ambition, themes, and emotional weight.
It has also received praise from many astronomers for its scientific accuracy and portrayal of theoretical astrophysics. Since its premiere, Interstellar gained a cult following,[5] and now is regarded by many sci-fi experts as one of the best science-fiction films of all time.
Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects, and received numerous other accolades"""

# Split into a list of sentences
texts = text.split('.')

# Clean up to remove empty spaces and new lines
texts = [t.strip(' \n') for t in texts]


model_path = "E:/huggingface/models/microsoft-Phi-3-mini-4k-instruct-GGUF-smashed/Phi-3-mini-4k-instruct.Q3_K_S.gguf"
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    max_tokens=500,
    n_ctx=2048,
    seed=42,
    verbose=False
)

# Embedding Model for converting text to numerical representations
embedding_model_path = "E:/huggingface/models/BAAI_bge-small-en-v1.5"
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_path
)

# Create a local vector database
db = FAISS.from_texts(texts, embedding_model)

# Create a prompt template
template = """<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>"""
prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# RAG Pipeline
rag = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db.as_retriever(),
    chain_type_kwargs={
        "prompt": prompt
    },
    verbose=True
)

result = rag.invoke('Interstellar premiered')
print(result)
```

```
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 6085.79it/s]


> Entering new RetrievalQA chain...

> Finished chain.
{'query': 'Interstellar premiered', 'result': " Interstellar premiered on October 26, 2rule-based systems:\n1. Identify relevant information related to the question in each document segment (in this case, only one).\nFor Document A and B combined knowledge pieces about 'Interstellar': It is a sci-fi film co-written/directed by Christopher Nolan with accolades for Best Visual Effects at Academy Awards. No other specific details from documents are provided on premiere date or time frame beyond the nomination year (2014).\n \nDocument C: The movie premiering in Los Angeles, California [relevant]. This indicates 'Interstellar' had its first showing here/regionally but doesn’t provide a specific event like awards nominations."}
```

&nbsp;

提升RAG的进阶技术：

1. 查询改写：当用户提问冗长或需要关联对话上下文时，使用LLM将原始查询转化为更利于检索的简洁形式。

2. 多查询RAG：对复杂问题生成多个关联查询。

3. 多跳RAG：针对需要分布推理的复杂问题，执行连续检索。

4. 查询路由：多数据源定向检索。根据不同问题查询不同的数据源。

5. 智能体RAG：数据源本身也可以抽象为LLM的工具。

&nbsp;

RAG效果评估：

1. 流畅性（fluency）：生成文本的流畅度和逻辑连贯性。

2. 感知效用（perceived utility）：回答内容的信息价值和实用价值

3. 引用召回率（citation recall）：外部事实陈述中获得完整引证支持的比例

4. 引用精确率（citation precision）：引用内容与相关论断支持的有效性

5. 忠实度（faithfulness）：答案与所提供上下文的一致性

6. 答案相关性（answer relevance）：答案与提问主题的契合度

&nbsp;

自动化评估框架：Ragas
https://www.ragas.io/


