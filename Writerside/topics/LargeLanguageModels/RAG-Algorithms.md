# RAG 算法优化：技术原理与前沿方法

检索增强生成 (RAG) 系统的性能很大程度上取决于其底层算法的有效性。本文从算法研究者的视角，深入探讨 RAG 系统的核心算法原理、优化方法和前沿研究方向。

## 嵌入模型优化 {id="embedding_optimization"}

嵌入模型是 RAG 系统的基础，其质量直接影响检索性能。

### 领域适应微调 {id="domain_adaptation"}

通用嵌入模型在特定领域（如医疗、法律、金融）的表现往往不尽如人意。领域适应微调可以显著提升模型在特定领域的性能。

#### 对比学习微调 {id="contrastive_learning"}

对比学习是优化嵌入空间的有效方法，通过拉近相似样本的距离，推远不相似样本的距离，提高嵌入的判别能力。

```python
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        # 归一化嵌入
        embeddings = F.normalize(embeddings, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建标签矩阵
        label_matrix = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # 移除对角线
        mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # 计算对比损失
        positives = torch.exp(similarity_matrix) * label_matrix
        negatives = torch.exp(similarity_matrix) * (1 - label_matrix)
        
        loss = -torch.log(
            positives.sum(dim=1) / (positives.sum(dim=1) + negatives.sum(dim=1))
        ).mean()
        
        return loss

def domain_finetune_embedding_model(base_model_name, train_data, output_path, epochs=10):
    """领域适应微调嵌入模型"""
    # 加载基础模型
    model = SentenceTransformer(base_model_name)
    
    # 准备训练数据
    train_examples = []
    for item in train_data:
        if "positive_pairs" in item:
            for text1, text2 in item["positive_pairs"]:
                train_examples.append(InputExample(texts=[text1, text2], label=1.0))
        if "hard_negatives" in item:
            for text, negatives in item["hard_negatives"]:
                for neg in negatives:
                    train_examples.append(InputExample(texts=[text, neg], label=0.0))
    
    # 创建数据加载器
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # 定义损失函数
    train_loss = losses.CosineSimilarityLoss(model)
    
    # 微调模型
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    # 保存微调后的模型
    model.save(output_path)
    
    return model
```

#### 指令微调 {id="instruction_tuning"}

指令微调通过明确的任务指令引导嵌入模型生成更适合检索的表示。

```python
def create_instruction_tuning_data(documents, queries, relevance_judgments):
    """创建指令微调数据"""
    instruction_data = []
    
    # 检索指令模板
    retrieval_instruction = "为检索任务生成文本嵌入。找到与查询'{query}'相关的文档。"
    
    # 为每个查询创建指令样本
    for query_id, query in queries.items():
        # 获取相关文档
        relevant_docs = [documents[doc_id] for doc_id in relevance_judgments.get(query_id, [])]
        
        if relevant_docs:
            # 创建指令样本
            instruction = retrieval_instruction.format(query=query)
            
            instruction_data.append({
                "instruction": instruction,
                "query": query,
                "positive_docs": relevant_docs,
                "negative_docs": sample_negative_docs(documents, relevance_judgments[query_id])
            })
    
    return instruction_data

def instruction_finetune_embedding_model(base_model, instruction_data, output_path):
    """指令微调嵌入模型"""
    # 实现指令微调逻辑
    # ...
    
    return finetuned_model
```

### 多任务学习 {id="multi_task_learning"}

多任务学习通过同时优化多个相关任务，提高嵌入模型的泛化能力和鲁棒性。

```python
def create_multitask_training_data(documents, queries, relevance_judgments):
    """创建多任务学习数据"""
    tasks = {
        "retrieval": [],  # 检索任务
        "clustering": [], # 聚类任务
        "classification": [], # 分类任务
        "similarity": []  # 相似度任务
    }
    
    # 为每个任务准备数据
    # ...
    
    return tasks

class MultitaskEmbeddingModel(nn.Module):
    """多任务嵌入模型"""
    def __init__(self, base_model_name, task_heads):
        super().__init__()
        # 加载基础编码器
        self.encoder = SentenceTransformer(base_model_name)
        
        # 任务特定头部
        self.task_heads = nn.ModuleDict()
        for task_name, head_config in task_heads.items():
            self.task_heads[task_name] = self._create_task_head(head_config)
    
    def _create_task_head(self, config):
        """创建任务特定头部"""
        layers = []
        input_dim = config.get("input_dim", self.encoder.get_sentence_embedding_dimension())
        
        for hidden_dim in config.get("hidden_dims", []):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, config["output_dim"]))
        
        if config.get("activation"):
            layers.append(config["activation"])
        
        return nn.Sequential(*layers)
    
    def forward(self, texts, task=None):
        """前向传播"""
        # 获取文本嵌入
        embeddings = self.encoder.encode(texts, convert_to_tensor=True)
        
        # 如果指定了任务，返回该任务的输出
        if task and task in self.task_heads:
            return self.task_heads[task](embeddings)
        
        # 否则返回基础嵌入
        return embeddings
```

## 检索算法优化 {id="retrieval_optimization"}

检索算法是 RAG 系统的核心，直接影响检索结果的相关性和多样性。

### 混合检索策略 {id="hybrid_retrieval"}

混合检索结合了多种检索方法的优势，提高检索的全面性和准确性。

#### 稠密-稀疏混合检索 {id="dense_sparse_hybrid"}

结合基于向量的稠密检索和基于关键词的稀疏检索：

```python
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.preprocessing import normalize

class HybridRetriever:
    """混合检索器"""
    def __init__(self, documents, embeddings, dense_weight=0.7):
        self.documents = documents
        self.embeddings = embeddings
        self.dense_weight = dense_weight
        
        # 准备稀疏检索
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # 归一化嵌入
        self.normalized_embeddings = normalize(embeddings)
    
    def retrieve(self, query, embedding_model, top_k=5):
        """执行混合检索"""
        # 稠密检索
        query_embedding = embedding_model.encode([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        dense_scores = np.dot(self.normalized_embeddings, query_embedding)
        dense_indices = np.argsort(dense_scores)[::-1][:top_k*2]
        dense_results = [(i, dense_scores[i]) for i in dense_indices]
        
        # 稀疏检索
        tokenized_query = query.lower().split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:top_k*2]
        sparse_results = [(i, sparse_scores[i]) for i in sparse_indices]
        
        # 归一化分数
        max_dense_score = max([score for _, score in dense_results]) if dense_results else 1.0
        max_sparse_score = max([score for _, score in sparse_results]) if sparse_results else 1.0
        
        normalized_dense = [(i, score/max_dense_score) for i, score in dense_results]
        normalized_sparse = [(i, score/max_sparse_score) for i, score in sparse_results]
        
        # 合并结果
        doc_scores = {}
        
        for i, score in normalized_dense:
            doc_scores[i] = self.dense_weight * score
            
        for i, score in normalized_sparse:
            if i in doc_scores:
                doc_scores[i] += (1 - self.dense_weight) * score
            else:
                doc_scores[i] = (1 - self.dense_weight) * score
        
        # 排序并返回结果
        sorted_indices = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)
        results = [(self.documents[i], doc_scores[i]) for i in sorted_indices[:top_k]]
        
        return results
```

#### 集成检索 {id="ensemble_retrieval"}

集成多个检索器的结果，通过投票或加权方式提高检索质量：

```python
class EnsembleRetriever:
    """集成检索器"""
    
    def __init__(self, retrievers, weights=None):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        
        # 归一化权重
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def retrieve(self, query, top_k=5):
        """执行集成检索"""
        # 收集所有检索器的结果
        all_results = []
        for i, retriever in enumerate(self.retrievers):
            results = retriever.retrieve(query, top_k=top_k*2)
            all_results.append((results, self.weights[i]))
        
        # 合并结果
        doc_scores = {}
        
        for results, weight in all_results:
            for doc, score in results:
                doc_id = self._get_doc_id(doc)
                if doc_id in doc_scores:
                    doc_scores[doc_id] = (doc_scores[doc_id][0] + score * weight, doc)
                else:
                    doc_scores[doc_id] = (score * weight, doc)
        
        # 排序并返回结果
        sorted_results = sorted(doc_scores.values(), key=lambda x: x[0], reverse=True)
        return [(doc, score) for score, doc in sorted_results[:top_k]]
    
    def _get_doc_id(self, doc):
        """获取文档唯一标识符"""
        if hasattr(doc, 'id'):
            return doc.id
        elif hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return doc.metadata['id']
        else:
            # 使用内容哈希作为标识符
            return hash(doc.page_content if hasattr(doc, 'page_content') else str(doc))
```

### 查询优化 {id="query_optimization"}

查询优化通过改进原始查询，提高检索的精确性和召回率。

#### 查询扩展 {id="query_expansion"}

通过添加相关术语或同义词扩展原始查询：

```python
def expand_query_with_synonyms(query, synonym_api):
    """使用同义词扩展查询"""
    # 分词
    tokens = query.split()
    
    # 为每个词查找同义词
    expanded_tokens = []
    for token in tokens:
        # 添加原始词
        expanded_tokens.append(token)
        
        # 查找同义词
        synonyms = synonym_api.get_synonyms(token, max_synonyms=2)
        expanded_tokens.extend(synonyms)
    
    # 构建扩展查询
    expanded_query = " ".join(expanded_tokens)
    return expanded_query

def expand_query_with_llm(query, llm):
    """使用大型语言模型扩展查询"""
    prompt = f"""请为以下搜索查询生成 5 个相关的关键词或短语，这些词应该能帮助扩展原始查询的语义范围。
    
    原始查询: {query}
    
    扩展关键词:"""
    
    response = llm.generate(prompt)
    keywords = [kw.strip() for kw in response.split('\n') if kw.strip()]
    
    # 合并原始查询和扩展关键词
    expanded_query = f"{query} {' '.join(keywords)}"
    return expanded_query
```

#### 查询重写 {id="query_rewriting"}

使用 LLM 重写查询，使其更适合检索：

```python
def rewrite_query_for_retrieval(query, llm):
    """重写查询以提高检索效果"""
    prompt = f"""请将以下用户查询重写为更适合文档检索的形式。
    添加可能相关的关键词，使用更精确的术语，但保持原始语义。
    不要添加不必要的限定词。
    
    原始查询: {query}
    
    重写后的查询:"""
    
    rewritten_query = llm.generate(prompt).strip()
    return rewritten_query

def generate_multiple_queries(query, llm, num_variations=3):
    """生成多个查询变体"""
    prompt = f"""请从不同角度重写以下查询，生成 {num_variations} 个不同的查询版本，以便更全面地检索相关信息。
    每个变体应该关注查询的不同方面或使用不同的表达方式。
    
    原始查询: {query}
    
    查询变体:"""
    
    response = llm.generate(prompt)
    query_variations = [q.strip() for q in response.split('\n') if q.strip()]
    
    # 确保返回指定数量的变体
    query_variations = query_variations[:num_variations]
    if len(query_variations) < num_variations:
        query_variations.extend([query] * (num_variations - len(query_variations)))
    
    return query_variations
```

### 重排序算法 {id="reranking_algorithms"}

重排序算法对初步检索结果进行精细排序，提高最终结果的相关性。

#### 交叉编码器重排序 {id="cross_encoder_reranking"}

使用交叉编码器模型评估查询-文档对的相关性：

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class CrossEncoderReranker:
    """交叉编码器重排序器"""
    
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
    
    def rerank(self, query, documents, top_k=None):
        """重排序文档"""
        if not documents:
            return []
        
        # 准备输入
        pairs = [(query, doc.page_content if hasattr(doc, 'page_content') else str(doc)) 
                for doc in documents]
        
        # 批处理评分
        batch_size = 32
        scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i+batch_size]
            
            # 编码
            features = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**features)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores)
        
        # 排序
        scored_documents = list(zip(documents, scores))
        scored_documents.sort(key=lambda x: x[1], reverse=True)
        
        # 返回结果
        if top_k:
            return [doc for doc, _ in scored_documents[:top_k]]
        return [doc for doc, _ in scored_documents]
```

#### 多阶段重排序 {id="multi_stage_reranking"}

通过多个阶段的重排序，平衡效率和精度：

```python
class MultiStageReranker:
    """多阶段重排序器"""
    
    def __init__(self, first_stage_k=100, second_stage_k=20, final_k=5):
        # 第一阶段：轻量级重排序（如 BM25 分数调整）
        self.first_stage_k = first_stage_k
        
        # 第二阶段：中等复杂度模型（如双塔模型）
        self.second_stage_k = second_stage_k
        self.bi_encoder = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-v4")
        
        # 第三阶段：高精度模型（如交叉编码器）
        self.final_k = final_k
        self.cross_encoder = CrossEncoderReranker()
    
    def rerank(self, query, initial_documents):
        """执行多阶段重排序"""
        # 第一阶段：保留前 N 个结果
        if len(initial_documents) > self.first_stage_k:
            first_stage_docs = initial_documents[:self.first_stage_k]
        else:
            first_stage_docs = initial_documents
        
        # 第二阶段：双塔模型重排序
        query_embedding = self.bi_encoder.encode(query, convert_to_tensor=True)
        doc_embeddings = self.bi_encoder.encode(
            [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in first_stage_docs],
            convert_to_tensor=True
        )
        
        # 计算相似度
        similarities = torch.matmul(doc_embeddings, query_embedding).cpu().numpy()
        
        # 排序
        second_stage_indices = np.argsort(similarities)[::-1][:self.second_stage_k]
        second_stage_docs = [first_stage_docs[i] for i in second_stage_indices]
        
        # 第三阶段：交叉编码器重排序
        final_docs = self.cross_encoder.rerank(query, second_stage_docs, self.final_k)
        
        return final_docs
```

#### 基于 LLM 的重排序 {id="llm_based_reranking"}

利用 LLM 的理解能力进行高质量重排序：

```python
def llm_rerank(query, documents, llm, top_k=5):
    """使用 LLM 重排序文档"""
    if not documents:
        return []
    
    # 准备评分提示
    scoring_template = """请评估以下文档与查询的相关性，给出 0-10 的分数。
    10 分表示文档完全回答了查询，0 分表示完全不相关。
    只返回分数，不要有其他文字。
    
    查询: {query}
    
    文档: {document}
    
    相关性分数 (0-10):"""
    
    # 为每个文档评分
    scores = []
    for doc in documents:
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        prompt = scoring_template.format(query=query, document=doc_text)
        
        try:
            # 获取 LLM 评分
            response = llm.generate(prompt)
            
            # 提取分数
            score = float(response.strip())
            scores.append(score)
        except:
            # 如果解析失败，给予默认低分
            scores.append(0.0)
    
    # 排序
    scored_documents = list(zip(documents, scores))
    scored_documents.sort(key=lambda x: x[1], reverse=True)
    
    # 返回结果
    return [doc for doc, _ in scored_documents[:top_k]]
```

## 上下文优化 {id="context_optimization"}

上下文优化关注如何处理和组织检索结果，以提供最有用的信息给生成模型。

### 上下文压缩 {id="context_compression"}

上下文压缩通过删除冗余和不相关信息，提高上下文质量和效率。

#### 基于 LLM 的上下文提取 {id="llm_based_extraction"}

使用 LLM 从检索文档中提取关键信息：

```python
def extract_relevant_context(query, documents, llm):
    """从检索文档中提取与查询相关的关键信息"""
    combined_text = "\n\n".join([
        doc.page_content if hasattr(doc, 'page_content') else str(doc)
        for doc in documents
    ])
    
    prompt = f"""以下是关于一个查询的多个文档片段。请提取与查询直接相关的关键信息，
    删除无关内容，但保留所有可能回答查询的重要事实和细节。
    
    查询: {query}
    
    文档内容:
    {combined_text}
    
    提取的关键信息:"""
    
    extracted_context = llm.generate(prompt)
    return extracted_context
```

#### 基于句子的上下文压缩 {id="sentence_based_compression"}

通过句子级别的相关性评估压缩上下文：

```python
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compress_context_by_sentences(query, documents, compression_ratio=0.5):
    """基于句子相关性压缩上下文"""
    # 确保 NLTK 资源可用
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # 提取所有句子
    all_sentences = []
    doc_boundaries = []
    current_position = 0
    
    for doc in documents:
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        sentences = sent_tokenize(doc_text)
        all_sentences.extend(sentences)
        
        # 记录文档边界
        current_position += len(sentences)
        doc_boundaries.append(current_position)
    
    # 如果句子太少，不进行压缩
    if len(all_sentences) <= 3:
        return "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in documents
        ])
    
    # 计算句子与查询的相关性
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(all_sentences + [query])
        query_vector = tfidf_matrix[-1]
        sentence_vectors = tfidf_matrix[:-1]
        
        # 计算相似度
        similarities = cosine_similarity(sentence_vectors, query_vector).flatten()
    except:
        # 如果向量化失败，使用简单的词重叠度量
        query_words = set(query.lower().split())
        similarities = []
        for sentence in all_sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words.intersection(sentence_words))
            similarities.append(overlap / max(1, len(query_words)))
    
    # 选择最相关的句子
    target_count = max(3, int(len(all_sentences) * compression_ratio))
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:target_count]
    top_indices = sorted(top_indices)  # 恢复原始顺序
    
    # 重建压缩后的文档
    compressed_docs = []
    start_idx = 0
    
    for end_idx in doc_boundaries:
        # 获取当前文档的句子索引
        doc_indices = [i for i in top_indices if start_idx <= i < end_idx]
        
        if doc_indices:
            # 重建文档
            doc_sentences = [all_sentences[i] for i in doc_indices]
            compressed_docs.append(" ".join(doc_sentences))
        
        start_idx = end_idx
    
    return "\n\n".join(compressed_docs)
```

### 信息融合 {id="information_fusion"}

信息融合通过合并和整合多个来源的信息，提供更全面的上下文。

#### 基于主题的信息聚合 {id="topic_based_aggregation"}

将检索结果按主题聚合，提供结构化上下文：

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def aggregate_by_topics(documents, n_topics=3):
    """将文档按主题聚合"""
    # 提取文档文本
    texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
    
    # 如果文档太少，不进行聚类
    if len(texts) <= n_topics:
        return [(f"Topic {i+1}", [documents[i]]) for i in range(len(documents))]
    
    # 向量化文档
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # 聚类
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # 提取每个聚类的关键词
    cluster_keywords = []
    for i in range(n_topics):
        cluster_docs = [texts[j] for j in range(len(texts)) if clusters[j] == i]
        if not cluster_docs:
            cluster_keywords.append("Miscellaneous")
            continue
            
        # 计算该聚类中每个词的平均 TF-IDF 值
        cluster_matrix = vectorizer.transform(cluster_docs)
        cluster_tfidf_avg = cluster_matrix.mean(axis=0).A1
        
        # 获取前 5 个关键词
        top_indices = cluster_tfidf_avg.argsort()[-5:][::-1]
        top_terms = [vectorizer.get_feature_names_out()[i] for i in top_indices]
        cluster_keywords.append(", ".join(top_terms))
    
    # 按主题组织文档
    topics = []
    for i in range(n_topics):
        topic_docs = [documents[j] for j in range(len(documents)) if clusters[j] == i]
        if topic_docs:
            topics.append((cluster_keywords[i], topic_docs))
    
    return topics

def create_structured_context(query, topics):
    """创建结构化上下文"""
    context = f"以下是关于查询 '{query}' 的相关信息，按主题组织：\n\n"
    
    for topic_name, docs in topics:
        context += f"## {topic_name}\n\n"
        
        for doc in docs:
            doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            context += f"- {doc_text}\n\n"
    
    return context
```

#### 冲突信息处理 {id="conflict_resolution"}

处理检索结果中的冲突信息，提高上下文一致性：

```python
def detect_and_resolve_conflicts(documents, llm):
    """检测并解决文档间的信息冲突"""
    # 提取文档文本
    texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
    combined_text = "\n\n".join(texts)
    
    prompt = """请分析以下文档片段，识别其中的信息冲突或不一致之处。
    对于每个冲突，请说明：
    1. 冲突的具体内容
    2. 涉及的不同说法
    3. 如果可能，提供解决冲突的方法或确定哪个信息更可靠
    
    文档内容:
    {text}
    
    冲突分析:"""
    
    # 获取冲突分析
    analysis = llm.generate(prompt.format(text=combined_text))
    
    # 如果发现冲突，添加冲突说明
    if "未发现冲突" not in analysis and "没有明显冲突" not in analysis:
        resolution_prompt = """基于以下冲突分析，请整合这些文档中的信息，
        解决冲突并提供一个一致的、准确的信息摘要。
        
        文档内容:
        {text}
        
        冲突分析:
        {analysis}
        
        整合后的信息:"""
        
        resolved_content = llm.generate(resolution_prompt.format(
            text=combined_text,
            analysis=analysis
        ))
        
        return resolved_content
    
    # 如果没有冲突，返回原始文档
    return combined_text
```

### 上下文排序 {id="context_ordering"}

上下文排序关注如何组织检索结果，以优化生成模型的理解和使用。

#### 相关性排序 {id="relevance_ordering"}

根据与查询的相关性对上下文进行排序：

```python
def order_by_relevance(query, documents, similarity_model):
    """根据相关性对文档排序"""
    # 计算每个文档与查询的相似度
    query_embedding = similarity_model.encode(query)
    
    scored_docs = []
    for doc in documents:
        doc_text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        doc_embedding = similarity_model.encode(doc_text)
        
        # 计算余弦相似度
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        scored_docs.append((doc, similarity))
    
    # 按相似度排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    return [doc for doc, _ in scored_docs]
```

#### 信息流排序 {id="information_flow_ordering"}

根据信息流逻辑对上下文进行排序，提高连贯性：

```python
def order_by_information_flow(documents, llm):
    """根据信息流逻辑对文档排序"""
    # 提取文档文本
    texts = [doc.page_content if hasattr(doc, 'page_content') else str(doc) for doc in documents]
    
    # 如果文档太少，不进行重排序
    if len(texts) <= 2:
        return documents
    
    # 为每个文档创建摘要
    summaries = []
    for text in texts:
        summary_prompt = f"请用一句话总结以下文本的主要内容：\n\n{text}\n\n总结："
        summary = llm.generate(summary_prompt).strip()
        summaries.append(summary)
    
    # 创建排序提示
    ordering_prompt = """请将以下文档摘要按照逻辑信息流排序，使其形成连贯的叙述。
    返回排序后的索引列表，索引从0开始。
    只返回索引数字，用逗号分隔，不要有其他文字。
    
    文档摘要:
    {summaries}
    
    排序索引:"""
    
    # 获取排序
    response = llm.generate(ordering_prompt.format(
        summaries="\n".join([f"{i}: {summary}" for i, summary in enumerate(summaries)])
    ))
    
    try:
        # 解析排序索引
        indices = [int(idx.strip()) for idx in response.split(',')]
        
        # 验证索引有效性
        if set(indices) != set(range(len(documents))):
            # 如果索引无效，返回原始顺序
            return documents
        
        # 按新顺序排序
        return [documents[i] for i in indices]
    except:
        # 如果解析失败，返回原始顺序
        return documents
```

## 动态 RAG 策略 {id="dynamic_rag_strategies"}

动态 RAG 策略根据查询特点和上下文情况，自适应地调整检索和生成过程。

### 查询路由 {id="query_routing"}

根据查询类型选择最合适的检索策略：

```python
class QueryRouter:
    """查询路由器"""
    
    def __init__(self, retrievers, llm):
        self.retrievers = retrievers
        self.llm = llm
    
    def route_query(self, query):
        """路由查询到合适的检索器"""
        # 分析查询类型
        analysis_prompt = """请分析以下查询，并确定其类型。
        选择以下类型之一：
        1. 事实型 - 寻求具体事实或数据
        2. 概念型 - 寻求概念解释或定义
        3. 程序型 - 寻求步骤或方法
        4. 比较型 - 寻求多个事物的比较
        5. 观点型 - 寻求观点或评价
        
        只返回类型编号，不要有其他文字。
        
        查询: {query}
        
        类型:"""
        
        response = self.llm.generate(analysis_prompt.format(query=query))
        
        try:
            query_type = int(response.strip())
        except:
            # 默认为事实型
            query_type = 1
        
        # 根据查询类型选择检索器
        if query_type == 1:  # 事实型
            return self.retrievers.get("factual", self.retrievers["default"])
        elif query_type == 2:  # 概念型
            return self.retrievers.get("conceptual", self.retrievers["default"])
        elif query_type == 3:  # 程序型
            return self.retrievers.get("procedural", self.retrievers["default"])
        elif query_type == 4:  # 比较型
            return self.retrievers.get("comparative", self.retrievers["default"])
        elif query_type == 5:  # 观点型
            return self.retrievers.get("opinion", self.retrievers["default"])
        else:
            return self.retrievers["default"]
```

### 自适应检索 {id="adaptive_retrieval"}

根据初步检索结果动态调整检索策略：

```python
class AdaptiveRetriever:
    """自适应检索器"""
    
    def __init__(self, base_retriever, fallback_retrievers, llm):
        self.base_retriever = base_retriever
        self.fallback_retrievers = fallback_retrievers
        self.llm = llm
    
    def retrieve(self, query, top_k=5):
        """执行自适应检索"""
        # 初始检索
        initial_results = self.base_retriever.retrieve(query, top_k)
        
        # 评估结果质量
        if not initial_results:
            # 如果没有结果，使用第一个备选检索器
            return self.fallback_retrievers[0].retrieve(query, top_k)
        
        # 评估结果相关性
        evaluation_prompt = """请评估以下检索结果与查询的相关性。
        给出 0-10 的分数，10 表示非常相关，0 表示完全不相关。
        只返回分数，不要有其他文字。
        
        查询: {query}
        
        检索结果:
        {results}
        
        相关性分数 (0-10):"""
        
        results_text = "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in initial_results[:2]  # 只评估前两个结果
        ])
        
        response = self.llm.generate(evaluation_prompt.format(
            query=query,
            results=results_text
        ))
        
        try:
            relevance_score = float(response.strip())
        except:
            relevance_score = 5.0  # 默认中等相关性
        
        # 根据相关性决定策略
        if relevance_score >= 7.0:
            # 结果相关性高，使用原始结果
            return initial_results
        elif relevance_score >= 4.0:
            # 结果相关性中等，尝试查询扩展
            expanded_query = expand_query_with_llm(query, self.llm)
            expanded_results = self.base_retriever.retrieve(expanded_query, top_k)
            
            # 合并结果并去重
            combined_results = self._merge_unique_results(initial_results, expanded_results)
            return combined_results[:top_k]
        else:
            # 结果相关性低，尝试不同的检索器
            alternative_results = []
            for retriever in self.fallback_retrievers:
                alt_results = retriever.retrieve(query, top_k)
                alternative_results.extend(alt_results)
            
            # 合并所有结果并去重
            all_results = self._merge_unique_results(initial_results, alternative_results)
            return all_results[:top_k]
    
    def _merge_unique_results(self, results1, results2):
        """合并结果并去重"""
        seen = set()
        unique_results = []
        
        for doc in results1 + results2:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                unique_results.append(doc)
        
        return unique_results
    
    def _get_doc_id(self, doc):
        """获取文档唯一标识符"""
        if hasattr(doc, 'id'):
            return doc.id
        elif hasattr(doc, 'metadata') and 'id' in doc.metadata:
            return doc.metadata['id']
        else:
            # 使用内容哈希作为标识符
            return hash(doc.page_content if hasattr(doc, 'page_content') else str(doc))
```

### 迭代检索 {id="iterative_retrieval"}

通过多轮检索迭代优化结果：

```python
def iterative_retrieval(query, retriever, llm, max_iterations=3):
    """执行迭代检索"""
    current_query = query
    best_results = []
    best_score = 0
    
    for i in range(max_iterations):
        # 当前查询的检索结果
        current_results = retriever.retrieve(current_query)
        
        if not current_results:
            continue
        
        # 评估结果
        evaluation_prompt = """请评估以下检索结果与原始查询的相关性。
        给出 0-10 的分数，10 表示完全回答了查询，0 表示完全不相关。
        只返回分数，不要有其他文字。
        
        原始查询: {original_query}
        当前查询: {current_query}
        
        检索结果:
        {results}
        
        相关性分数 (0-10):"""
        
        results_text = "\n\n".join([
            doc.page_content if hasattr(doc, 'page_content') else str(doc)
            for doc in current_results[:3]  # 只评估前三个结果
        ])
        
        response = self.llm.generate(evaluation_prompt.format(
            original_query=query,
            current_query=current_query,
            results=results_text
        ))
        
        try:
            current_score = float(response.strip())
        except:
            current_score = 0.0
        
        # 更新最佳结果
        if current_score > best_score:
            best_score = current_score
            best_results = current_results
        
        # 如果分数足够高，提前结束
        if best_score >= 8.0:
            break
        
        # 生成改进的查询
        refinement_prompt = """基于原始查询和当前检索结果，请生成一个改进的查询，
        以获取更相关的信息。新查询应该更具体，并解决当前结果中的不足。
        
        原始查询: {original_query}
        当前查询: {current_query}
        
        当前检索结果:
        {results}
        
        改进的查询:"""
        
        current_query = self.llm.generate(refinement_prompt.format(
            original_query=query,
            current_query=current_query,
            results=results_text
        )).strip()
    
    return best_results
```

## 评估与优化框架 {id="evaluation_framework"}

评估与优化框架用于系统性地测试和改进 RAG 系统的性能。

### 离线评估 {id="offline_evaluation"}

通过预定义的测试集评估 RAG 系统性能：

```python
def evaluate_rag_system(rag_system, test_queries, ground_truth, metrics=None):
    """评估 RAG 系统性能"""
    if metrics is None:
        metrics = ["precision", "recall", "f1", "ndcg"]
    
    results = {metric: [] for metric in metrics}
    
    for query_id, query in test_queries.items():
        # 获取系统响应
        retrieved_docs = rag_system.retrieve(query)
        generated_answer = rag_system.generate(query, retrieved_docs)
        
        # 获取参考答案
        reference_docs = ground_truth["relevant_docs"].get(query_id, [])
        reference_answer = ground_truth["answers"].get(query_id, "")
        
        # 计算检索指标
        if "precision" in metrics:
            precision = calculate_precision(retrieved_docs, reference_docs)
            results["precision"].append(precision)
        
        if "recall" in metrics:
            recall = calculate_recall(retrieved_docs, reference_docs)
            results["recall"].append(recall)
        
        if "f1" in metrics:
            if "precision" in locals() and "recall" in locals():
                f1 = calculate_f1(precision, recall)
            else:
                f1 = calculate_f1_score(retrieved_docs, reference_docs)
            results["f1"].append(f1)
        
        if "ndcg" in metrics:
            ndcg = calculate_ndcg(retrieved_docs, reference_docs)
            results["ndcg"].append(ndcg)
        
        # 计算生成指标
        if "rouge" in metrics:
            rouge = calculate_rouge(generated_answer, reference_answer)
            results["rouge"].append(rouge)
        
        if "bleu" in metrics:
            bleu = calculate_bleu(generated_answer, reference_answer)
            results["bleu"].append(bleu)
        
        if "bertscore" in metrics:
            bertscore = calculate_bertscore(generated_answer, reference_answer)
            results["bertscore"].append(bertscore)
    
    # 计算平均指标
    avg_results = {metric: sum(scores)/len(scores) if scores else 0 
                  for metric, scores in results.items()}
    
    return avg_results
```

### 在线评估 {id="online_evaluation"}

通过用户反馈评估和改进 RAG 系统：

```python
class OnlineEvaluator:
    """在线评估器"""
    
    def __init__(self, rag_system, feedback_store):
        self.rag_system = rag_system
        self.feedback_store = feedback_store
    
    def collect_feedback(self, query, retrieved_docs, generated_answer, user_feedback):
        """收集用户反馈"""
        feedback_data = {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "generated_answer": generated_answer,
            "user_feedback": user_feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback_store.add_feedback(feedback_data)
    
    def analyze_feedback(self, time_period=None):
        """分析用户反馈"""
        # 获取指定时间段的反馈
        feedback_data = self.feedback_store.get_feedback(time_period)
        
        if not feedback_data:
            return {"message": "No feedback data available for analysis"}
        
        # 分析检索性能
        retrieval_ratings = [data["user_feedback"].get("retrieval_rating", 0) 
                            for data in feedback_data 
                            if "retrieval_rating" in data["user_feedback"]]
        
        # 分析生成性能
        generation_ratings = [data["user_feedback"].get("generation_rating", 0) 
                             for data in feedback_data 
                             if "generation_rating" in data["user_feedback"]]
        
        # 分析整体满意度
        satisfaction_ratings = [data["user_feedback"].get("satisfaction", 0) 
                               for data in feedback_data 
                               if "satisfaction" in data["user_feedback"]]
        
        # 计算平均评分
        avg_retrieval = sum(retrieval_ratings) / len(retrieval_ratings) if retrieval_ratings else 0
        avg_generation = sum(generation_ratings) / len(generation_ratings) if generation_ratings else 0
        avg_satisfaction = sum(satisfaction_ratings) / len(satisfaction_ratings) if satisfaction_ratings else 0
        
        # 识别常见问题
        common_issues = self._identify_common_issues(feedback_data)
        
        return {
            "avg_retrieval_rating": avg_retrieval,
            "avg_generation_rating": avg_generation,
            "avg_satisfaction": avg_satisfaction,
            "common_issues": common_issues,
            "feedback_count": len(feedback_data)
        }
    
    def _identify_common_issues(self, feedback_data):
        """识别常见问题"""
        # 提取负面反馈
        negative_feedback = [data for data in feedback_data 
                            if data["user_feedback"].get("satisfaction", 5) < 3]
        
        if not negative_feedback:
            return []
        
        # 提取问题描述
        issue_descriptions = [data["user_feedback"].get("comments", "") 
                             for data in negative_feedback 
                             if "comments" in data["user_feedback"]]
        
        # 使用简单的频率分析识别常见问题
        issue_keywords = {
            "irrelevant": 0,
            "incomplete": 0,
            "inaccurate": 0,
            "outdated": 0,
            "missing context": 0,
            "hallucination": 0
        }
        
        for description in issue_descriptions:
            for keyword in issue_keywords:
                if keyword in description.lower():
                    issue_keywords[keyword] += 1
        
        # 返回频率最高的问题
        sorted_issues = sorted(issue_keywords.items(), key=lambda x: x[1], reverse=True)
        return [{"issue": issue, "count": count} for issue, count in sorted_issues if count > 0]
```

### 持续优化 {id="continuous_optimization"}

通过持续学习和适应改进 RAG 系统：

```python
class RAGOptimizer:
    """RAG 系统优化器"""
    
    def __init__(self, rag_system, evaluator, optimization_config):
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.config = optimization_config
        self.optimization_history = []
    
    def optimize(self, test_queries=None, user_feedback=None):
        """优化 RAG 系统"""
        # 评估当前性能
        current_performance = self._evaluate_performance(test_queries, user_feedback)
        
        # 记录初始性能
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": current_performance,
            "changes": []
        })
        
        # 确定优化目标
        optimization_targets = self._identify_optimization_targets(current_performance)
        
        # 应用优化策略
        changes_made = []
        for target in optimization_targets:
            change = self._apply_optimization_strategy(target)
            if change:
                changes_made.append(change)
        
        # 重新评估性能
        new_performance = self._evaluate_performance(test_queries, user_feedback)
        
        # 记录优化结果
        self.optimization_history.append({
            "timestamp": datetime.now().isoformat(),
            "performance": new_performance,
            "changes": changes_made
        })
        
        # 计算改进
        improvement = {
            metric: new_performance.get(metric, 0) - current_performance.get(metric, 0)
            for metric in new_performance
        }
        
        return {
            "initial_performance": current_performance,
            "optimized_performance": new_performance,
            "improvement": improvement,
            "changes_made": changes_made
        }
    
    def _evaluate_performance(self, test_queries=None, user_feedback=None):
        """评估系统性能"""
        performance = {}
        
        # 离线评估
        if test_queries:
            offline_metrics = self.evaluator.evaluate_offline(
                self.rag_system, test_queries
            )
            performance.update(offline_metrics)
        
        # 在线评估
        if user_feedback:
            online_metrics = self.evaluator.analyze_feedback(user_feedback)
            performance.update(online_metrics)
        
        return performance
    
    def _identify_optimization_targets(self, performance):
        """识别需要优化的目标"""
        targets = []
        
        # 检查检索性能
        if performance.get("precision", 1.0) < self.config["thresholds"]["precision"]:
            targets.append("retrieval_precision")
        
        if performance.get("recall", 1.0) < self.config["thresholds"]["recall"]:
            targets.append("retrieval_recall")
        
        # 检查生成性能
        if performance.get("answer_relevance", 1.0) < self.config["thresholds"]["answer_relevance"]:
            targets.append("generation_relevance")
        
        if performance.get("answer_accuracy", 1.0) < self.config["thresholds"]["answer_accuracy"]:
            targets.append("generation_accuracy")
        
        # 检查用户满意度
        if performance.get("user_satisfaction", 5.0) < self.config["thresholds"]["user_satisfaction"]:
            targets.append("user_satisfaction")
        
        return targets
    
    def _apply_optimization_strategy(self, target):
        """应用优化策略"""
        strategies = {
            "retrieval_precision": self._optimize_retrieval_precision,
            "retrieval_recall": self._optimize_retrieval_recall,
            "generation_relevance": self._optimize_generation_relevance,
            "generation_accuracy": self._optimize_generation_accuracy,
            "user_satisfaction": self._optimize_user_satisfaction
        }
        
        if target in strategies:
            return strategies[target]()
        
        return None
    
    def _optimize_retrieval_precision(self):
        """优化检索精确度"""
        # 实现检索精确度优化策略
        # 例如：调整相似度阈值、改进重排序算法等
        
        # 返回所做的更改
        return {
            "target": "retrieval_precision",
            "strategy": "adjusted_similarity_threshold",
            "details": "Increased similarity threshold from 0.7 to 0.8"
        }
    
    def _optimize_retrieval_recall(self):
        """优化检索召回率"""
        # 实现检索召回率优化策略
        # 例如：启用查询扩展、增加检索数量等
        
        # 返回所做的更改
        return {
            "target": "retrieval_recall",
            "strategy": "enabled_query_expansion",
            "details": "Enabled semantic query expansion with LLM"
        }
    
    def _optimize_generation_relevance(self):
        """优化生成相关性"""
        # 实现生成相关性优化策略
        # 例如：改进上下文选择、调整提示模板等
        
        # 返回所做的更改
        return {
            "target": "generation_relevance",
            "strategy": "improved_context_selection",
            "details": "Implemented relevance-based context ordering"
        }
    
    def _optimize_generation_accuracy(self):
        """优化生成准确性"""
        # 实现生成准确性优化策略
        # 例如：添加事实验证、改进冲突解决等
        
        # 返回所做的更改
        return {
            "target": "generation_accuracy",
            "strategy": "added_fact_verification",
            "details": "Added post-generation fact verification step"
        }
    
    def _optimize_user_satisfaction(self):
        """优化用户满意度"""
        # 实现用户满意度优化策略
        # 例如：改进响应格式、添加引用等
        
        # 返回所做的更改
        return {
            "target": "user_satisfaction",
            "strategy": "improved_response_format",
            "details": "Added source citations and confidence indicators"
        }
```

## 结论与未来方向 {id="conclusion"}

RAG 算法优化是一个快速发展的领域，随着大型语言模型和检索技术的进步，我们可以期待更多创新方法的出现。本文介绍的技术原理和方法为研究者和工程师提供了系统性的框架，用于构建和优化高性能的 RAG 系统。

未来的研究方向包括：

1. **多模态 RAG**：扩展 RAG 系统以处理图像、音频和视频等多模态数据
2. **个性化 RAG**：根据用户偏好和历史交互自适应调整检索和生成策略
3. **长上下文 RAG**：优化处理长文档和大量检索结果的能力
4. **知识图谱增强 RAG**：结合结构化知识图谱提高检索精度和推理能力
5. **自监督优化 RAG**：通过自监督学习持续改进 RAG 系统性能

随着这些方向的发展，RAG 系统将变得更加智能、高效和可靠，为各种应用场景提供更优质的信息检索和生成服务。

## 参考资料 {id="references"}

1. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.
2. Gao, L., et al. (2023). Precise Zero-shot Dense Retrieval without Relevance Labels. ACL 2023.
3. Izacard, G., et al. (2022). Atlas: Few-shot Learning with Retrieval Augmented Language Models. JMLR 2022.
4. Khattab, O., et al. (2022). Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-intensive NLP. ArXiv.
5. Shi, W., et al. (2023). Replug: Retrieval-Augmented Black-Box Language Models. ArXiv.
6. Ram, P., et al. (2023). In-Context Retrieval-Augmented Language Models. TACL 2023.
7. Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ArXiv.
8. Jiang, Z., et al. (2023). Active Retrieval Augmented Generation. ArXiv.
9. Trivedi, H., et al. (2022). Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. ACL 2023.
10. Wang, S., et al. (2023). Enhancing Retrieval-Augmented Large Language Models with Iterative Retrieval-Generation Synergy. ArXiv.

## 附录：RAG 系统实现示例 {id="appendix"}

以下是一个简化的 RAG 系统实现示例，集成了本文讨论的多种优化技术：

```python
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple, Optional

class OptimizedRAGSystem:
    """优化的 RAG 系统"""
    
    def __init__(
        self,
        documents: List[Dict[str, Any]],
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "gpt2-xl",  # 实际应用中可替换为更强大的模型
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        hybrid_search_weight: float = 0.7,
        use_query_optimization: bool = True,
        use_context_compression: bool = True,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 5
    ):
        # 初始化文档存储
        self.documents = documents
        self.doc_texts = [doc["content"] for doc in documents]
        self.doc_metadata = [doc.get("metadata", {}) for doc in documents]
        
        # 初始化模型
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        
        # 配置参数
        self.hybrid_weight = hybrid_search_weight
        self.use_query_optimization = use_query_optimization
        self.use_context_compression = use_context_compression
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        
        # 预计算文档嵌入
        self.doc_embeddings = self._compute_embeddings(self.doc_texts)
        
        # 初始化 BM25 (稀疏检索)
        self._initialize_sparse_retrieval()
        
        # 初始化重排序器
        if reranker_model_name:
            self._initialize_reranker(reranker_model_name)
        else:
            self.reranker = None
    
    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """计算文本嵌入"""
        return self.embedding_model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
    
    def _initialize_sparse_retrieval(self):
        """初始化稀疏检索"""
        from rank_bm25 import BM25Okapi
        
        # 分词
        tokenized_docs = [doc.lower().split() for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _initialize_reranker(self, model_name: str):
        """初始化重排序器"""
        from transformers import AutoModelForSequenceClassification
        
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reranker = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # 移至 GPU (如果可用)
        if torch.cuda.is_available():
            self.reranker = self.reranker.to("cuda")
    
    def optimize_query(self, query: str) -> str:
        """优化查询"""
        # 在实际应用中，这里可以使用 LLM 进行查询重写
        # 简化实现：添加同义词扩展
        tokens = query.lower().split()
        synonyms = {
            "good": ["excellent", "great"],
            "bad": ["poor", "terrible"],
            "big": ["large", "huge"],
            "small": ["tiny", "little"]
            # 实际应用中可使用更完整的同义词库或 WordNet
        }
        
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            if token in synonyms:
                expanded_tokens.append(synonyms[token][0])  # 只添加第一个同义词，避免查询过长
        
        return " ".join(expanded_tokens)
    
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        # 查询优化
        if self.use_query_optimization:
            optimized_query = self.optimize_query(query)
        else:
            optimized_query = query
        
        # 混合检索
        results = self._hybrid_search(optimized_query, self.top_k_retrieval)
        
        # 重排序
        if self.reranker:
            results = self._rerank_results(query, results)
        
        return results[:self.top_k_rerank]
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """执行混合检索"""
        # 稠密检索
        query_embedding = self.embedding_model.encode(query)
        dense_scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # 稀疏检索
        tokenized_query = query.lower().split()
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # 归一化分数
        if np.max(dense_scores) > 0:
            dense_scores = dense_scores / np.max(dense_scores)
        if np.max(sparse_scores) > 0:
            sparse_scores = sparse_scores / np.max(sparse_scores)
        
        # 混合分数
        hybrid_scores = self.hybrid_weight * dense_scores + (1 - self.hybrid_weight) * sparse_scores
        
        # 获取前 top_k 结果
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        # 构建结果
        results = []
        for idx in top_indices:
            results.append({
                "content": self.doc_texts[idx],
                "metadata": self.doc_metadata[idx],
                "score": float(hybrid_scores[idx])
            })
        
        return results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序检索结果"""
        if not results:
            return []
        
        # 准备输入对
        pairs = [(query, doc["content"]) for doc in results]
        
        # 批处理评分
        features = self.reranker_tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        # 移至 GPU (如果可用)
        if torch.cuda.is_available():
            features = {k: v.to("cuda") for k, v in features.items()}
        
        # 预测
        with torch.no_grad():
            scores = self.reranker(**features).logits.squeeze(-1).cpu().numpy()
        
        # 更新分数
        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
        
        # 重新排序
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        return results
    
    def compress_context(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """压缩上下文"""
        # 在实际应用中，这里可以使用更复杂的上下文压缩方法
        # 简化实现：基于 TF-IDF 相似度选择最相关的句子
        
        import nltk
        from nltk.tokenize import sent_tokenize
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # 确保 NLTK 资源可用
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # 提取所有句子
        all_sentences = []
        doc_boundaries = []
        current_position = 0
        
        for doc in documents:
            sentences = sent_tokenize(doc["content"])
            all_sentences.extend(sentences)
            
            # 记录文档边界
            current_position += len(sentences)
            doc_boundaries.append(current_position)
        
        # 如果句子太少，不进行压缩
        if len(all_sentences) <= 5:
            return "\n\n".join([doc["content"] for doc in documents])
        
        # 计算句子与查询的相关性
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(all_sentences + [query])
            query_vector = tfidf_matrix[-1]
            sentence_vectors = tfidf_matrix[:-1]
            
            # 计算相似度
            similarities = cosine_similarity(sentence_vectors, query_vector).flatten()
        except:
            # 如果向量化失败，使用简单的词重叠度量
            query_words = set(query.lower().split())
            similarities = []
            for sentence in all_sentences:
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                similarities.append(overlap / max(1, len(query_words)))
        
        # 选择最相关的句子
        compression_ratio = 0.5  # 压缩率
        target_count = max(5, int(len(all_sentences) * compression_ratio))
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:target_count]
        top_indices = sorted(top_indices)  # 恢复原始顺序
        
        # 重建压缩后的文档
        compressed_docs = []
        start_idx = 0
        
        for end_idx in doc_boundaries:
            # 获取当前文档的句子索引
            doc_indices = [i for i in top_indices if start_idx <= i < end_idx]
            
            if doc_indices:
                # 重建文档
                doc_sentences = [all_sentences[i] for i in doc_indices]
                compressed_docs.append(" ".join(doc_sentences))
            
            start_idx = end_idx
        
        return "\n\n".join(compressed_docs)
    
    def generate(self, query: str, context: Optional[str] = None) -> str:
        """生成回答"""
        # 如果没有提供上下文，先检索
        if context is None:
            retrieved_docs = self.retrieve(query)
            
            if not retrieved_docs:
                return "抱歉，我没有找到相关信息。"
            
            # 压缩上下文
            if self.use_context_compression:
                context = self.compress_context(query, retrieved_docs)
            else:
                context = "\n\n".join([doc["content"] for doc in retrieved_docs])
        
        # 构建提示
        prompt = f"""请根据以下信息回答问题。如果信息不足以回答问题，请说明无法回答。

信息:
{context}

问题: {query}

回答:"""
        
        # 编码提示
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 移至 GPU (如果可用)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            self.llm = self.llm.to("cuda")
        
        # 生成回答
        outputs = self.llm.generate(
            **inputs,
            max_length=len(inputs["input_ids"][0]) + 200,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # 解码回答
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        answer = generated_text.split("回答:")[-1].strip()
        
        return answer
    
    def answer_question(self, query: str) -> Dict[str, Any]:
        """回答问题并返回详细信息"""
        # 检索
        retrieved_docs = self.retrieve(query)
        
        # 准备上下文
        if self.use_context_compression and retrieved_docs:
            context = self.compress_context(query, retrieved_docs)
        else:
            context = "\n\n".join([doc["content"] for doc in retrieved_docs]) if retrieved_docs else ""
        
        # 生成回答
        answer = self.generate(query, context)
        
        # 返回结果
        return {
            "query": query,
            "answer": answer,
            "sources": [
                {
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": doc.get("rerank_score", doc["score"])
                }
                for doc in retrieved_docs
            ],
            "context_used": context
        }
```

这个示例实现了一个优化的 RAG 系统，包含了混合检索、查询优化、重排序和上下文压缩等技术。在实际应用中，可以根据具体需求进一步扩展和优化这个系统。
