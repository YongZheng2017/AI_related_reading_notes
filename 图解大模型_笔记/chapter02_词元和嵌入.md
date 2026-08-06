# 词元和嵌入

分词：把文本分解成更小的片段（词或词的一部分）  

嵌入向量：将词元转换为数值表示，捕捉其含义。 

&nbsp;

## LLM的分词

openai的在线分词器：https://platform.openai.com/tokenizer 

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

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"

# Tokenize the input prompt

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

# Generate the text

generation_output = model.generate(
  input_ids=input_ids,
  max_new_tokens=50
)

# Print the output

print(tokenizer.decode(generation_output[0]))

print(input_ids)
for id in input_ids:
    print(tokenizer.decode(id))
print(generation_output)
for id in generation_output:
    print(tokenizer.decode(id))
```

&nbsp;

分词方法：找到一组尽可能高效的词元来表示文本数据集。

- 字节对编码（BEF，byte pair encoding）：广泛用于GPT模型

- WordPiece：用于BERT模型。
  
  &nbsp;

分词方式：

1. 词级分词：按单词分词。早期的word2vec等模型中很常见，NLP中使用的少，推荐系统等NLP外的场景中有使用。
   
   挑战：可能无法处理分词器训练完成之后才出现在数据集中的新词。

2. 子词级分词：包含词和词的一部分，能够表示新词。

3. 字符级分词：按字符分词。建模比较困难。

4. 字节级分词：将词元分解为unicode编码的单个字节。
   
   &nbsp;

训练好的LLM分词器

1. BERT基座模型：
   a.分词方法：WordPiece

2. GPT：
   a.分词方法：BPE
   
   &nbsp;

分词器属性：

1. 分词方法：BPE是最流行的一种。  

2. 用于初始化分词器的参数：  
   a. 词表大小，常见的有30K，50K，100K。  
   b. 特殊词元：文本开始词元、文本结束词元、填充词元（填充模型输入中未使用的位  置）、未知词元、CLS词元（分类词元，主要用于分类任务）、掩码词元  
   c. 大小写处理策略

3. 数据领域：训练所使用的数据集
   
     &nbsp;

## 词元嵌入

预训练模型与其分词器是绑定的。语言模型会创建与上下文相关的词嵌入。  
文本嵌入：用单个向量表示长度超过一个词元的文本片段。用于分类、语义搜索和RAG等场景。

&nbsp;
word2vec：

```
import gensim.downloader as api

# Download embeddings (66MB, glove, trained on wikipedia, vector size: 50)

# Other options include "word2vec-google-news-300"

# More options at https://github.com/RaRe-Technologies/gensim-data

model = api.load("glove-wiki-gigaword-50")
result = model.most_similar([model['king']], topn=11)
print(result)
```

&nbsp;

训练：使用滑动窗口生成训练文本，嵌入向量通过分类任务生成，该分类任务训练神经网络，以预测词是否经常出现在相同的上下文中。  
skip-gram：选择相邻词的方法。  
负采样（negative sampling）：通过从数据集中随机采样来添加负例。

&nbsp;

**接收两个向量并预测它们是否具有某种关系的模型思想**，是机器学习中最强大的思想之一，在语言模型中屡试不爽，也是连接文本和图像等不同模态的核心。
