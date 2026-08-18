# 文本聚类

## 文本聚类的通用流程

步骤：

1. 使用嵌入模型将输入文档转换为嵌入向量

2. 使用降维模型将嵌入向量降至更低维度空间

3. 使用聚类模型将降维后的向量进行聚类

&nbsp;

嵌入向量是高维数据，对聚类技术来说比较困难，可通过降维来解决；  
降维是一种压缩技术，是为了帮助聚类模型更高效地创建有意义的簇。   
常用的降维方法：

1. 主成分分析（Principal Component Analysis, PCA）

2. 统一流形逼近

3. 投影（Uniform Manifold Approximation and Projection, UMAP）  

&nbsp;

聚类：

1. 基于质心的算法：k-means，需要生成一组预设的簇

2. 基于密度的算法：HDBSCAN（具有噪声的分层密度空间聚类），是DBSCAN的层次化变体。不需要预设簇，不会强制所有数据点属于某个簇。

&nbsp;

数据集：MaartenGr/arxiv_nlp  
下载数据集：hf download MaartenGr/arxiv_nlp --repo-type dataset --local-dir E:\huggingface\datasets\arxiv_nlp  

&nbsp;

模型：thenlper/gte-small  
下载模型：modelscope download --model AkaCoder404/gte-small --repo-type model --local-dir E:\huggingface\models\thenlper_gte-small

```
# pip install bertopic datasets openai datamapplot
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np


# Load data from huggingface
dataset = load_dataset("E:/huggingface/datasets/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
model_path = "E:/huggingface/models/thenlper_gte-small"
embedding_model = SentenceTransformer(model_path)
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
print(embeddings.shape)

# Reduce the input embeddings from 384 dimenions to 5 dimenions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric='cosine', random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate?
print(len(set(clusters)))

# Print first three documents in cluster 0
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")
```

```
Batches: 100%|██████████| 1405/1405 [02:26<00:00,  9.59it/s]
(44949, 384)
165
  This works aims to design a statistical machine translation from English text
to American Sign Language (ASL). The system is based on Moses tool with some
modifications and the results are synthesized through a 3D avatar for
interpretation. First, we translate the input text to gloss, a written fo... 

  Researches on signed languages still strongly dissociate lin- guistic issues
related on phonological and phonetic aspects, and gesture studies for
recognition and synthesis purposes. This paper focuses on the imbrication of
motion and meaning for the analysis, synthesis and evaluation of sign lang... 

  Modern computational linguistic software cannot produce important aspects of
sign language translation. Using some researches we deduce that the majority of
automatic sign language translation systems ignore many aspects when they
generate animation; therefore the interpretation lost the truth inf... 
```

&nbsp;

## 主题建模

主题建模：在文本数据集合中寻找主题或潜在语义。传统上，主题通过若干关键词来表示。 
经典方法：狄利克雷分配（latent Dirichlet allocation, LDA），假设每个主题都由语料库词表中词的概率分布来表示。  
这些经典方法通常使用词袋技术提取文本数据的主要特征，而没有考虑词和短语的上下文及含义；而文本聚类因基于Transformer嵌入向量，通过注意力机制针对语义相似性和上下文含义进行了优化。  
通过文本聚类和主题建模框架BERTopic，将文本聚类扩展到主题建模。  
BERTopic：一个模块化主题建模框架。处理流程可以分为两部分。

1. 第一部分：聚类。首先对文档进行嵌入，降维，最后对降维的嵌入向量进行聚类；其次利用词袋方法，对词表中的词在语料库中的分布建模。

2. 第二部分：主题表示。计算词在类中的权重。

```
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
import pandas as pd


# Load data from huggingface
dataset = load_dataset("E:/huggingface/datasets/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Create an embedding for each abstract
model_path = "E:/huggingface/models/thenlper_gte-small"
embedding_model = SentenceTransformer(model_path)
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
print(embeddings.shape)

# Reduce the input embeddings from 384 dimenions to 5 dimenions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric='cosine', random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

# Fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# Train our model with our previously defined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

print(topic_model.get_topic_info())



# Save original representations
from copy import deepcopy
original_topics = deepcopy(topic_model.topic_representations_)


def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]

    return df

from bertopic.representation import KeyBERTInspired

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))
```

```
Batches: 100%|██████████| 1405/1405 [02:28<00:00,  9.45it/s]
(44949, 384)
2026-08-18 14:36:44,911 - BERTopic - Dimensionality - Fitting the dimensionality reduction algorithm
2026-08-18 14:37:12,356 - BERTopic - Dimensionality - Completed ✓
2026-08-18 14:37:12,357 - BERTopic - Cluster - Start clustering the reduced embeddings
2026-08-18 14:37:13,417 - BERTopic - Cluster - Completed ✓
2026-08-18 14:37:13,423 - BERTopic - Representation - Fine-tuning topics using representation models.
2026-08-18 14:37:17,135 - BERTopic - Representation - Completed ✓
     Topic  ...                                Representative_Docs
0       -1  ...  [  Generative Large Language Models (LLMs) hav...
1        0  ...  [  We introduce Wav2Seq, the first self-superv...
2        1  ...  [  Medical code assignment from clinical text ...
3        2  ...  [  Neural network-based approaches have become...
4        3  ...  [  Sentence-level relation extraction mainly a...
..     ...  ...                                                ...
160    159  ...  [  While there has been significant progress t...
161    160  ...  [  Prompt optimization aims to find the best p...
162    161  ...  [  The usage of more than one language in the ...
163    162  ...  [  The Mixture of Experts (MoE) models are an ...
164    163  ...  [  Conversations have an intrinsic one-to-many...

[165 rows x 5 columns]
   Topic  ...                                            Updated
0      0  ...  transcription | translation | language | speec...
1      1  ...         nlp | clinical | annotated | corpus | text
2      2  ...  summarization | summarizers | summaries | summ...
3      3  ...  relations | relation | relational | sentences ...
4      4  ...         entity | entities | name | labeled | named

[5 rows x 3 columns]
```
