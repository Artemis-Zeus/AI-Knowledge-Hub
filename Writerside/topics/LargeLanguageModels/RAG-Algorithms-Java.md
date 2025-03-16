# RAG 算法 Java 实现：框架选择与代码示例

Java 生态系统提供了多种工具和框架，可用于构建高效的检索增强生成 (RAG) 系统。本文将重点介绍 Java 环境下 RAG 系统的实现方法、可用框架以及代码示例，帮助开发者在 Java 平台上构建高性能的 RAG 应用。

## Java RAG 框架生态 {id="java_rag_ecosystem"}

Java 平台上有多个成熟的框架可用于构建 RAG 系统的各个组件：

### 向量检索框架 {id="vector_search_frameworks"}

1. **Lucene/Elasticsearch**：Apache Lucene 及其分布式搜索引擎 Elasticsearch 提供了强大的文本检索能力，最新版本已支持向量搜索。
2. **Milvus-Java**：Milvus 向量数据库的 Java 客户端，专为大规模向量相似度搜索设计。
3. **Weaviate Java Client**：Weaviate 向量搜索引擎的 Java 客户端。
4. **Qdrant Java Client**：Qdrant 向量数据库的 Java 客户端。
5. **Vespa**：雅虎开发的搜索引擎，支持向量搜索和混合检索。

### 嵌入模型框架 {id="embedding_frameworks"}

1. **DJL (Deep Java Library)**：Amazon 开发的深度学习库，支持多种模型格式和推理引擎。
2. **Tribuo**：Oracle 开发的机器学习库，提供了多种算法实现。
3. **ND4J/DL4J**：用于 Java 的科学计算库和深度学习框架。
4. **HuggingFace Java Client**：通过 API 调用 HuggingFace 模型的客户端。

### 文本处理框架 {id="text_processing_frameworks"}

1. **OpenNLP**：Apache 开发的自然语言处理库。
2. **Stanford CoreNLP**：斯坦福大学开发的综合 NLP 工具包。
3. **NLTK4J**：NLTK 的 Java 移植版。
4. **Langchain4j**：LangChain 的 Java 实现，专为 LLM 应用设计。

## 嵌入模型实现 {id="embedding_implementation"}

### 使用 DJL 加载嵌入模型 {id="djl_embedding"}

DJL 是一个强大的 Java 深度学习库，可以加载和使用各种预训练模型：

<code-block collapsible="true" lang="java">
package com.example.rag.embedding;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.nlp.embedding.TextEmbedding;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DJLEmbeddingModel implements EmbeddingModel {
    private ZooModel<String, float[]> model;
    private Predictor<String, float[]> predictor;

    public DJLEmbeddingModel(String modelPath) throws ModelNotFoundException, MalformedModelException, IOException {
        // 创建模型加载条件
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelPath(modelPath)
                .optProgress(new ProgressBar())
                .build();

        // 加载模型
        model = ModelZoo.loadModel(criteria);
        predictor = model.newPredictor();
    }

    @Override
    public float[] encode(String text) throws TranslateException {
        return predictor.predict(text);
    }

    @Override
    public List<float[]> batchEncode(List<String> texts) throws TranslateException {
        List<float[]> embeddings = new ArrayList<>();
        for (String text : texts) {
            embeddings.add(encode(text));
        }
        return embeddings;
    }

    @Override
    public void close() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }
}
</code-block>

### 使用 Langchain4j 的嵌入功能 {id="langchain4j_embedding"}

Langchain4j 提供了与多种嵌入服务的集成：

<code-block collapsible="true" lang="java">
package com.example.rag.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;

import java.util.ArrayList;
import java.util.List;

public class Langchain4jEmbeddingModel implements com.example.rag.embedding.EmbeddingModel {
    private final EmbeddingModel embeddingModel;

    public Langchain4jEmbeddingModel() {
        // 使用本地 AllMiniLmL6V2 模型
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
    }

    @Override
    public float[] encode(String text) {
        Embedding embedding = embeddingModel.embed(text).content();
        return convertToFloatArray(embedding);
    }

    @Override
    public List<float[]> batchEncode(List<String> texts) {
        List<Embedding> embeddings = embeddingModel.embedAll(texts).content();
        List<float[]> result = new ArrayList<>();
        
        for (Embedding embedding : embeddings) {
            result.add(convertToFloatArray(embedding));
        }
        
        return result;
    }

    private float[] convertToFloatArray(Embedding embedding) {
        List<Double> vector = embedding.vectorAsList();
        float[] result = new float[vector.size()];
        
        for (int i = 0; i < vector.size(); i++) {
            result[i] = vector.get(i).floatValue();
        }
        
        return result;
    }

    @Override
    public void close() {
        // Langchain4j 模型不需要显式关闭
    }
}
</code-block>

## 检索算法实现 {id="retrieval_implementation"}

### 使用 Lucene 实现混合检索 {id="lucene_hybrid_retrieval"}

Lucene 是 Java 生态中最强大的搜索引擎库，可以实现高效的混合检索：

<code-block collapsible="true" lang="java">
package com.example.rag.retrieval;

import com.example.rag.document.Document;
import com.example.rag.embedding.EmbeddingModel;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.document.KnnVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

public class LuceneHybridRetriever implements Retriever {
    private static final String CONTENT_FIELD = "content";
    private static final String VECTOR_FIELD = "vector";
    private static final String ID_FIELD = "id";
    
    private final Directory directory;
    private final EmbeddingModel embeddingModel;
    private final float hybridWeight;
    private final int dimensions;
    
    public LuceneHybridRetriever(Path indexPath, EmbeddingModel embeddingModel, float hybridWeight, int dimensions) throws IOException {
        this.directory = FSDirectory.open(indexPath);
        this.embeddingModel = embeddingModel;
        this.hybridWeight = hybridWeight;
        this.dimensions = dimensions;
    }
    
    public void indexDocuments(List<Document> documents) throws Exception {
        IndexWriterConfig config = new IndexWriterConfig(new StandardAnalyzer());
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (Document doc : documents) {
                org.apache.lucene.document.Document luceneDoc = new org.apache.lucene.document.Document();
                
                // 添加文本字段
                luceneDoc.add(new TextField(CONTENT_FIELD, doc.getContent(), Field.Store.YES));
                luceneDoc.add(new StoredField(ID_FIELD, doc.getId()));
                
                // 添加向量字段
                float[] embedding = embeddingModel.encode(doc.getContent());
                luceneDoc.add(new KnnVectorField(VECTOR_FIELD, embedding, VectorSimilarityFunction.COSINE));
                
                writer.addDocument(luceneDoc);
            }
        }
    }
    
    @Override
    public List<Document> retrieve(String query, int topK) throws Exception {
        try (DirectoryReader reader = DirectoryReader.open(directory)) {
            IndexSearcher searcher = new IndexSearcher(reader);
            
            // 稀疏检索（BM25）
            Query textQuery = new TermQuery(new Term(CONTENT_FIELD, query));
            TopDocs textResults = searcher.search(textQuery, topK * 2);
            
            // 稠密检索（向量相似度）
            float[] queryEmbedding = embeddingModel.encode(query);
            KnnVectorQuery vectorQuery = new KnnVectorQuery(VECTOR_FIELD, queryEmbedding, topK * 2);
            TopDocs vectorResults = searcher.search(vectorQuery, topK * 2);
            
            // 混合结果
            return hybridSearch(searcher, textResults, vectorResults, topK);
        }
    }
    
    private List<Document> hybridSearch(IndexSearcher searcher, TopDocs textResults, 
                                        TopDocs vectorResults, int topK) throws IOException {
        // 创建文档ID到分数的映射
        java.util.Map<Integer, Float> docScores = new java.util.HashMap<>();
        
        // 归一化文本搜索分数
        float maxTextScore = 0.0f;
        for (ScoreDoc scoreDoc : textResults.scoreDocs) {
            maxTextScore = Math.max(maxTextScore, scoreDoc.score);
        }
        
        // 添加归一化后的文本搜索分数
        if (maxTextScore > 0) {
            for (ScoreDoc scoreDoc : textResults.scoreDocs) {
                float normalizedScore = scoreDoc.score / maxTextScore * (1 - hybridWeight);
                docScores.put(scoreDoc.doc, normalizedScore);
            }
        }
        
        // 归一化向量搜索分数
        float maxVectorScore = 0.0f;
        for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
            maxVectorScore = Math.max(maxVectorScore, scoreDoc.score);
        }
        
        // 添加归一化后的向量搜索分数
        if (maxVectorScore > 0) {
            for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
                float normalizedScore = scoreDoc.score / maxVectorScore * hybridWeight;
                docScores.merge(scoreDoc.doc, normalizedScore, Float::sum);
            }
        }
        
        // 排序并获取前 topK 个结果
        List<java.util.Map.Entry<Integer, Float>> sortedEntries = new ArrayList<>(docScores.entrySet());
        sortedEntries.sort((e1, e2) -> Float.compare(e2.getValue(), e1.getValue()));
        
        List<Document> results = new ArrayList<>();
        for (int i = 0; i < Math.min(topK, sortedEntries.size()); i++) {
            int docId = sortedEntries.get(i).getKey();
            org.apache.lucene.document.Document luceneDoc = searcher.doc(docId);
            
            Document doc = new Document();
            doc.setId(luceneDoc.get(ID_FIELD));
            doc.setContent(luceneDoc.get(CONTENT_FIELD));
            doc.setScore(sortedEntries.get(i).getValue());
            
            results.add(doc);
        }
        
        return results;
    }
    
    @Override
    public void close() throws IOException {
        directory.close();
    }
}
</code-block>

### 使用 Elasticsearch 实现混合检索 {id="elasticsearch_hybrid_retrieval"}

Elasticsearch 提供了分布式搜索能力，适合大规模 RAG 系统：

<code-block collapsible="true" lang="java">
package com.example.rag.retrieval;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.elasticsearch.core.SearchResponse;
import co.elastic.clients.elasticsearch.core.search.Hit;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.ElasticsearchTransport;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import com.example.rag.document.Document;
import com.example.rag.embedding.EmbeddingModel;
import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class ElasticsearchHybridRetriever implements Retriever {
    private final ElasticsearchClient client;
    private final String indexName;
    private final EmbeddingModel embeddingModel;
    private final float hybridWeight;
    
    public ElasticsearchHybridRetriever(String hostname, int port, String indexName, 
                                        EmbeddingModel embeddingModel, float hybridWeight) {
        // 创建低级客户端
        RestClient restClient = RestClient.builder(
                new HttpHost(hostname, port)).build();
        
        // 创建传输层
        ElasticsearchTransport transport = new RestClientTransport(
                restClient, new JacksonJsonpMapper());
        
        // 创建 API 客户端
        this.client = new ElasticsearchClient(transport);
        this.indexName = indexName;
        this.embeddingModel = embeddingModel;
        this.hybridWeight = hybridWeight;
    }
    
    public void indexDocuments(List<Document> documents) throws IOException {
        for (Document doc : documents) {
            // 计算文档嵌入
            float[] embedding = embeddingModel.encode(doc.getContent());
            
            // 创建索引请求
            client.index(i -> i
                    .index(indexName)
                    .id(doc.getId())
                    .document(Map.of(
                            "content", doc.getContent(),
                            "embedding", embedding
                    ))
            );
        }
        
        // 刷新索引
        client.indices().refresh(r -> r.index(indexName));
    }
    
    @Override
    public List<Document> retrieve(String query, int topK) throws Exception {
        // 计算查询嵌入
        float[] queryEmbedding = embeddingModel.encode(query);
        
        // 执行混合搜索
        SearchResponse<Map> response = client.search(s -> s
                .index(indexName)
                .size(topK)
                .query(q -> q
                        .bool(b -> b
                                // 文本相似度查询 (BM25)
                                .should(sh -> sh
                                        .match(m -> m
                                                .field("content")
                                                .query(query)
                                                .boost(1 - hybridWeight)
                                        )
                                )
                                // 向量相似度查询
                                .should(sh -> sh
                                        .scriptScore(ss -> ss
                                                .query(sq -> sq.matchAll(ma -> ma))
                                                .script(sc -> sc
                                                        .source("cosineSimilarity(params.query_vector, 'embedding') + 1.0")
                                                        .params("query_vector", queryEmbedding)
                                                )
                                                .boost(hybridWeight)
                                        )
                                )
                        )
                ),
                Map.class
        );
        
        // 处理结果
        List<Document> results = new ArrayList<>();
        for (Hit<Map> hit : response.hits().hits()) {
            Document doc = new Document();
            doc.setId(hit.id());
            doc.setContent((String) hit.source().get("content"));
            doc.setScore((float) hit.score());
            results.add(doc);
        }
        
        return results;
    }
    
    @Override
    public void close() throws IOException {
        // 关闭客户端
    }
}
</code-block>

### 使用 Langchain4j 实现检索 {id="langchain4j_retrieval"}

Langchain4j 提供了内置的检索功能，简化了 RAG 系统的实现：

<code-block collapsible="true" lang="java">
package com.example.rag.retrieval;

import com.example.rag.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.ArrayList;
import java.util.List;

public class Langchain4jRetriever implements Retriever {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final DocumentSplitter documentSplitter;
    
    public Langchain4jRetriever() {
        // 初始化嵌入模型
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        
        // 初始化嵌入存储
        this.embeddingStore = new InMemoryEmbeddingStore<>();
        
        // 初始化文档分割器
        this.documentSplitter = DocumentSplitters.recursive(300, 0);
    }
    
    public void indexDocuments(List<Document> documents) {
        // 转换为 Langchain4j 文档
        List<dev.langchain4j.data.document.Document> langchainDocs = new ArrayList<>();
        for (Document doc : documents) {
            langchainDocs.add(dev.langchain4j.data.document.Document.from(
                    doc.getContent(),
                    dev.langchain4j.data.document.Metadata.from("id", doc.getId())
            ));
        }
        
        // 使用 EmbeddingStoreIngestor 处理文档
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(documentSplitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        
        ingestor.ingest(langchainDocs);
    }
    
    @Override
    public List<Document> retrieve(String query, int topK) {
        // 使用嵌入模型和存储进行检索
        List<TextSegment> relevantSegments = embeddingStore.findRelevant(
                embeddingModel.embed(query).content(),
                topK
        );
        
        // 转换为文档
        List<Document> results = new ArrayList<>();
        for (TextSegment segment : relevantSegments) {
            Document doc = new Document();
            doc.setId(segment.metadata().get("id"));
            doc.setContent(segment.text());
            doc.setScore(segment.score());
            results.add(doc);
        }
        
        return results;
    }
    
    @Override
    public void close() {
        // 不需要显式关闭
    }
}
</code-block>

## 重排序算法实现 {id="reranking_implementation"}

### 使用 DJL 实现交叉编码器重排序 {id="djl_cross_encoder"}

使用 DJL 加载交叉编码器模型进行重排序：

<code-block collapsible="true" lang="java">
package com.example.rag.reranking;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.example.rag.document.Document;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class DJLCrossEncoderReranker implements Reranker {
    private ZooModel<String[], Float> model;
    private Predictor<String[], Float> predictor;
    
    public DJLCrossEncoderReranker(String modelPath) throws ModelNotFoundException, MalformedModelException, IOException {
        // 创建模型加载条件
        Criteria<String[], Float> criteria = Criteria.builder()
                .setTypes(String[].class, Float.class)
                .optModelPath(modelPath)
                .optProgress(new ProgressBar())
                .build();
        
        // 加载模型
        model = ModelZoo.loadModel(criteria);
        predictor = model.newPredictor();
    }
    
    @Override
    public List<Document> rerank(String query, List<Document> documents, int topK) throws TranslateException {
        List<Document> rerankedDocs = new ArrayList<>(documents);
        
        // 为每个文档计算相关性分数
        for (Document doc : rerankedDocs) {
            String[] input = new String[]{query, doc.getContent()};
            float score = predictor.predict(input);
            doc.setScore(score);
        }
        
        // 按分数排序
        rerankedDocs.sort(Comparator.comparing(Document::getScore).reversed());
        
        // 返回前 topK 个结果
        return rerankedDocs.subList(0, Math.min(topK, rerankedDocs.size()));
    }
    
    @Override
    public void close() {
        if (predictor != null) {
            predictor.close();
        }
        if (model != null) {
            model.close();
        }
    }
}
</code-block>

### 使用 Langchain4j 实现 LLM 重排序 {id="langchain4j_llm_reranking"}

使用 Langchain4j 的 LLM 集成进行重排序：

<code-block collapsible="true" lang="java">
package com.example.rag.reranking;

import com.example.rag.document.Document;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

public class Langchain4jLLMReranker implements Reranker {
    private final ChatLanguageModel llm;
    private final ExecutorService executor;
    
    public Langchain4jLLMReranker(String apiKey) {
        this.llm = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.0)
                .build();
        
        this.executor = Executors.newFixedThreadPool(5);
    }
    
    @Override
    public List<Document> rerank(String query, List<Document> documents, int topK) {
        List<Document> rerankedDocs = new ArrayList<>(documents);
        
        // 并行评分
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (Document doc : rerankedDocs) {
            CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                String prompt = String.format(
                        "请评估以下文档与查询的相关性，给出 0-10 的分数。只返回分数，不要有其他文字。\n\n" +
                        "查询: %s\n\n" +
                        "文档: %s\n\n" +
                        "相关性分数 (0-10):",
                        query, doc.getContent()
                );
                
                String response = llm.generate(prompt);
                
                try {
                    float score = Float.parseFloat(response.trim());
                    doc.setScore(score);
                } catch (NumberFormatException e) {
                    // 如果解析失败，给予默认低分
                    doc.setScore(0.0f);
                }
            }, executor);
            
            futures.add(future);
        }
        
        // 等待所有评分完成
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        
        // 按分数排序
        rerankedDocs.sort(Comparator.comparing(Document::getScore).reversed());
        
        // 返回前 topK 个结果
        return rerankedDocs.subList(0, Math.min(topK, rerankedDocs.size()));
    }
    
    @Override
    public void close() {
        executor.shutdown();
    }
}
</code-block>

## 上下文优化实现 {id="context_optimization_implementation"}

### 使用 OpenNLP 实现基于句子的上下文压缩 {id="opennlp_context_compression"}

使用 OpenNLP 进行句子分割和相关性评估：

<code-block collapsible="true" lang="java">
package com.example.rag.context;

import com.example.rag.document.Document;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class OpenNLPContextCompressor implements ContextProcessor {
    private final SentenceDetectorME sentenceDetector;
    private final double compressionRatio;
    
    public OpenNLPContextCompressor(String modelPath, double compressionRatio) throws IOException {
        try (InputStream modelIn = new FileInputStream(modelPath)) {
            SentenceModel model = new SentenceModel(modelIn);
            this.sentenceDetector = new SentenceDetectorME(model);
        }
        this.compressionRatio = compressionRatio;
    }
    
    @Override
    public String process(String query, List<Document> documents) {
        // 提取所有句子
        List<String> allSentences = new ArrayList<>();
        List<Integer> docBoundaries = new ArrayList<>();
        int currentPosition = 0;
        
        for (Document doc : documents) {
            String[] sentences = sentenceDetector.sentDetect(doc.getContent());
            allSentences.addAll(Arrays.asList(sentences));
            
            // 记录文档边界
            currentPosition += sentences.length;
            docBoundaries.add(currentPosition);
        }
        
        // 如果句子太少，不进行压缩
        if (allSentences.size() <= 3) {
            return documents.stream()
                    .map(Document::getContent)
                    .collect(Collectors.joining("\n\n"));
        }
        
        // 计算每个句子与查询的相似度
        List<SentenceScore> sentenceScores = new ArrayList<>();
        for (int i = 0; i < allSentences.size(); i++) {
            String sentence = allSentences.get(i);
            // 简单的词重叠相似度计算
            double score = calculateOverlapScore(query, sentence);
            sentenceScores.add(new SentenceScore(i, sentence, score));
        }
        
        // 按相似度排序
        sentenceScores.sort(Comparator.comparing(SentenceScore::getScore).reversed());
        
        // 选择前 N% 的句子
        int numToKeep = (int) Math.ceil(allSentences.size() * compressionRatio);
        List<SentenceScore> selectedSentences = sentenceScores.subList(0, numToKeep);
        
        // 按原始顺序排序
        selectedSentences.sort(Comparator.comparing(SentenceScore::getIndex));
        
        // 重建文本，保持文档边界
        StringBuilder result = new StringBuilder();
        int lastDocIndex = 0;
        
        for (SentenceScore sentenceScore : selectedSentences) {
            int docIndex = 0;
            for (int i = 0; i < docBoundaries.size(); i++) {
                if (sentenceScore.getIndex() < docBoundaries.get(i)) {
                    docIndex = i;
                    break;
                }
            }
            
            // 如果是新文档，添加分隔符
            if (docIndex > lastDocIndex) {
                result.append("\n\n");
                lastDocIndex = docIndex;
            }
            
            result.append(sentenceScore.getSentence()).append(" ");
        }
        
        return result.toString().trim();
    }
    
    private double calculateOverlapScore(String query, String sentence) {
        // 分词
        String[] queryTokens = query.toLowerCase().split("\\s+");
        String[] sentenceTokens = sentence.toLowerCase().split("\\s+");
        
        // 计算重叠词数
        int overlap = 0;
        for (String queryToken : queryTokens) {
            for (String sentenceToken : sentenceTokens) {
                if (queryToken.equals(sentenceToken)) {
                    overlap++;
                    break;
                }
            }
        }
        
        // 归一化分数
        return (double) overlap / queryTokens.length;
    }
    
    private static class SentenceScore {
        private final int index;
        private final String sentence;
        private final double score;
        
        public SentenceScore(int index, String sentence, double score) {
            this.index = index;
            this.sentence = sentence;
            this.score = score;
        }
        
        public int getIndex() {
            return index;
        }
        
        public String getSentence() {
            return sentence;
        }
        
        public double getScore() {
            return score;
        }
    }
    
    @Override
    public void close() {
        // OpenNLP 模型不需要显式关闭
    }
}
</code-block>

### 使用 Langchain4j 实现 LLM 上下文压缩 {id="langchain4j_context_compression"}

使用 Langchain4j 的 LLM 集成进行上下文压缩：

<code-block collapsible="true" lang="java">
package com.example.rag.context;

import com.example.rag.document.Document;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.util.List;
import java.util.stream.Collectors;

public class Langchain4jLLMCompressor implements ContextProcessor {
    private final ChatLanguageModel llm;
    private final int maxTokens;
    
    public Langchain4jLLMCompressor(String apiKey, int maxTokens) {
        this.llm = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.0)
                .build();
        
        this.maxTokens = maxTokens;
    }
    
    @Override
    public String process(String query, List<Document> documents) {
        // 合并文档内容
        String allContent = documents.stream()
                .map(Document::getContent)
                .collect(Collectors.joining("\n\n"));
        
        // 如果内容较短，不进行压缩
        if (allContent.length() < 1000) {
            return allContent;
        }
        
        // 构建压缩提示
        String prompt = String.format(
                "我需要你帮我压缩以下文本，使其更加简洁但保留与查询相关的所有重要信息。\n\n" +
                "查询: %s\n\n" +
                "文本内容:\n%s\n\n" +
                "请提供压缩后的文本，确保保留与查询相关的所有关键信息。不要添加任何解释，只返回压缩后的文本。",
                query, allContent
        );
        
        // 使用 LLM 压缩文本
        String compressedContent = llm.generate(prompt);
        
        return compressedContent;
    }
    
    @Override
    public void close() {
        // 不需要显式关闭
    }
}
</code-block>

## 完整 RAG 系统实现 {id="complete_rag_implementation"}

### 基于 Langchain4j 的完整 RAG 系统 {id="langchain4j_rag_system"}

使用 Langchain4j 构建完整的 RAG 系统：

<code-block collapsible="true" lang="java">
package com.example.rag;

import com.example.rag.document.Document;
import dev.langchain4j.chain.ConversationalRetrievalChain;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.retriever.EmbeddingStoreRetriever;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.util.ArrayList;
import java.util.List;

public class Langchain4jRAGSystem {
    private final EmbeddingModel embeddingModel;
    private final EmbeddingStore<TextSegment> embeddingStore;
    private final ChatLanguageModel chatModel;
    private final ConversationalRetrievalChain chain;
    private final ChatMemory chatMemory;
    
    public Langchain4jRAGSystem(String openAiApiKey) {
        // 初始化嵌入模型
        this.embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        
        // 初始化嵌入存储
        this.embeddingStore = new InMemoryEmbeddingStore<>();
        
        // 初始化聊天模型
        this.chatModel = OpenAiChatModel.builder()
                .apiKey(openAiApiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.7)
                .build();
        
        // 初始化检索器
        EmbeddingStoreRetriever retriever = EmbeddingStoreRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(5)
                .minScore(0.6)
                .build();
        
        // 初始化聊天记忆
        this.chatMemory = MessageWindowChatMemory.builder()
                .maxMessages(10)
                .build();
        
        // 初始化 RAG 链
        this.chain = ConversationalRetrievalChain.builder()
                .chatLanguageModel(chatModel)
                .retriever(retriever)
                .chatMemory(chatMemory)
                .build();
    }
    
    public void indexDocuments(List<Document> documents) {
        // 转换为 Langchain4j 文档
        List<dev.langchain4j.data.document.Document> langchainDocs = new ArrayList<>();
        for (Document doc : documents) {
            langchainDocs.add(dev.langchain4j.data.document.Document.from(
                    doc.getContent(),
                    dev.langchain4j.data.document.Metadata.from("id", doc.getId())
            ));
        }
        
        // 文档分割器
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 0);
        
        // 使用 EmbeddingStoreIngestor 处理文档
        EmbeddingStoreIngestor ingestor = EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build();
        
        ingestor.ingest(langchainDocs);
    }
    
    public String query(String userQuery) {
        // 使用 RAG 链处理查询
        AiMessage response = chain.execute(UserMessage.from(userQuery));
        return response.text();
    }
    
    public void close() {
        // 不需要显式关闭
    }
}
</code-block>

### 基于 Lucene 和 DJL 的自定义 RAG 系统 {id="custom_rag_system"}

使用 Lucene 和 DJL 构建自定义 RAG 系统：

<code-block collapsible="true" lang="java">
package com.example.rag;

import ai.djl.MalformedModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;
import com.example.rag.context.ContextProcessor;
import com.example.rag.context.OpenNLPContextCompressor;
import com.example.rag.document.Document;
import com.example.rag.embedding.DJLEmbeddingModel;
import com.example.rag.embedding.EmbeddingModel;
import com.example.rag.reranking.DJLCrossEncoderReranker;
import com.example.rag.reranking.Reranker;
import com.example.rag.retrieval.LuceneHybridRetriever;
import com.example.rag.retrieval.Retriever;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

public class CustomRAGSystem implements AutoCloseable {
    private final EmbeddingModel embeddingModel;
    private final Retriever retriever;
    private final Reranker reranker;
    private final ContextProcessor contextProcessor;
    private final ChatLanguageModel llm;
    
    public CustomRAGSystem(String embeddingModelPath, Path indexPath, String crossEncoderPath, 
                          String sentenceModelPath, String openAiApiKey) 
            throws ModelNotFoundException, MalformedModelException, IOException {
        
        // 初始化嵌入模型
        this.embeddingModel = new DJLEmbeddingModel(embeddingModelPath);
        
        // 初始化检索器
        this.retriever = new LuceneHybridRetriever(indexPath, embeddingModel, 0.7f, 384);
        
        // 初始化重排序器
        this.reranker = new DJLCrossEncoderReranker(crossEncoderPath);
        
        // 初始化上下文处理器
        this.contextProcessor = new OpenNLPContextCompressor(sentenceModelPath, 0.6);
        
        // 初始化 LLM
        this.llm = OpenAiChatModel.builder()
                .apiKey(openAiApiKey)
                .modelName("gpt-3.5-turbo")
                .temperature(0.7)
                .build();
    }
    
    public void indexDocuments(List<Document> documents) throws Exception {
        ((LuceneHybridRetriever) retriever).indexDocuments(documents);
    }
    
    public String query(String userQuery) throws Exception {
        // 1. 检索相关文档
        List<Document> retrievedDocs = retriever.retrieve(userQuery, 10);
        
        // 2. 重排序文档
        List<Document> rerankedDocs = reranker.rerank(userQuery, retrievedDocs, 5);
        
        // 3. 优化上下文
        String context = contextProcessor.process(userQuery, rerankedDocs);
        
        // 4. 构建提示
        String prompt = String.format(
                "请基于以下上下文回答问题。如果上下文中没有足够信息，请说明你不知道，不要编造答案。\n\n" +
                "上下文:\n%s\n\n" +
                "问题: %s",
                context, userQuery
        );
        
        // 5. 生成回答
        String response = llm.generate(prompt);
        
        return response;
    }
    
    @Override
    public void close() throws Exception {
        embeddingModel.close();
        retriever.close();
        reranker.close();
        contextProcessor.close();
    }
}
</code-block>

## 实用工具类 {id="utility_classes"}

### 文档类 {id="document_class"}

用于表示文档的基本类：

<code-block collapsible="true" lang="java">
package com.example.rag.document;

public class Document {
    private String id;
    private String content;
    private float score;
    
    public Document() {
    }
    
    public Document(String id, String content) {
        this.id = id;
        this.content = content;
    }
    
    public String getId() {
        return id;
    }
    
    public void setId(String id) {
        this.id = id;
    }
    
    public String getContent() {
        return content;
    }
    
    public void setContent(String content) {
        this.content = content;
    }
    
    public float getScore() {
        return score;
    }
    
    public void setScore(float score) {
        this.score = score;
    }
}
</code-block>

### 嵌入模型接口 {id="embedding_model_interface"}

定义嵌入模型的通用接口：

<code-block collapsible="true" lang="java">
package com.example.rag.embedding;

import ai.djl.translate.TranslateException;

import java.util.List;

public interface EmbeddingModel extends AutoCloseable {
    float[] encode(String text) throws TranslateException;
    List<float[]> batchEncode(List<String> texts) throws TranslateException;
}
</code-block>

### 检索器接口 {id="retriever_interface"}

定义检索器的通用接口：

<code-block collapsible="true" lang="java">
package com.example.rag.retrieval;

import com.example.rag.document.Document;

import java.util.List;

public interface Retriever extends AutoCloseable {
    List<Document> retrieve(String query, int topK) throws Exception;
}
</code-block>

### 重排序器接口 {id="reranker_interface"}

定义重排序器的通用接口：

<code-block collapsible="true" lang="java">
package com.example.rag.reranking;

import com.example.rag.document.Document;

import java.util.List;

public interface Reranker extends AutoCloseable {
    List<Document> rerank(String query, List<Document> documents, int topK) throws Exception;
}
</code-block>

### 上下文处理器接口 {id="context_processor_interface"}

定义上下文处理器的通用接口：

<code-block collapsible="true" lang="java">
package com.example.rag.context;

import com.example.rag.document.Document;

import java.util.List;

public interface ContextProcessor extends AutoCloseable {
    String process(String query, List<Document> documents) throws Exception;
}
</code-block>

## 示例应用 {id="example_application"}

### 简单的 RAG 命令行应用 {id="rag_cli_application"}

一个简单的命令行 RAG 应用示例：

<code-block collapsible="true" lang="java">
package com.example.rag.app;

import com.example.rag.Langchain4jRAGSystem;
import com.example.rag.document.Document;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

public class RAGCliApplication {
    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("用法: java RAGCliApplication <OpenAI API Key> <文档目录>");
            System.exit(1);
        }
        
        String apiKey = args[0];
        String docsDir = args[1];
        
        try {
            // 初始化 RAG 系统
            Langchain4jRAGSystem ragSystem = new Langchain4jRAGSystem(apiKey);
            
            // 加载文档
            List<Document> documents = loadDocuments(docsDir);
            System.out.println("加载了 " + documents.size() + " 个文档");
            
            // 索引文档
            ragSystem.indexDocuments(documents);
            System.out.println("文档索引完成");
            
            // 交互式查询
            BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
            
            while (true) {
                System.out.print("\n请输入查询 (输入 'exit' 退出): ");
                String query = reader.readLine().trim();
                
                if ("exit".equalsIgnoreCase(query)) {
                    break;
                }
                
                // 处理查询
                String response = ragSystem.query(query);
                System.out.println("\n回答: " + response);
            }
            
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static List<Document> loadDocuments(String docsDir) throws IOException {
        List<Document> documents = new ArrayList<>();
        Path dir = Paths.get(docsDir);
        
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException(docsDir + " 不是一个有效的目录");
        }
        
        Files.list(dir)
                .filter(path -> path.toString().endsWith(".txt"))
                .forEach(path -> {
                    try {
                        String content = Files.readString(path);
                        String id = UUID.randomUUID().toString();
                        documents.add(new Document(id, content));
                    } catch (IOException e) {
                        System.err.println("无法读取文件 " + path + ": " + e.getMessage());
                    }
                });
        
        return documents;
    }
}
</code-block>

## 总结 {id="conclusion"}

Java 生态系统提供了丰富的工具和框架，可以构建高效的 RAG 系统。本文介绍了使用 Java 实现 RAG 系统的各个组件，包括嵌入模型、检索算法、重排序算法和上下文优化等。

主要框架和工具包括：

1. **DJL (Deep Java Library)**：用于加载和使用深度学习模型，包括嵌入模型和交叉编码器。
2. **Lucene/Elasticsearch**：用于高效的文本和向量检索。
3. **OpenNLP**：用于文本处理和句子分割。
4. **Langchain4j**：Java 版的 LangChain，简化了 RAG 系统的构建。

通过组合这些工具和框架，开发者可以在 Java 平台上构建功能强大、性能优异的 RAG 系统，满足各种应用场景的需求。
