from ..tool import Tool, ToolParameter
from Core.llm import CoreLLM
from Memory.embedding import get_text_embedder, get_dimension
from Memory.Storage.qdrantDB import QdrantVectorStore
from .rag_util.rag_ingestion import RagIngestion
from .rag_util.rag_retriever import RagRetriever
import typing
from typing import Optional, Dict, Any, List, Union
import os
import hashlib
import uuid
import re
import time

class Rag(Tool):
    def __init__(self,
                knowledgeBaseURL:str = "./knowledge_database",
                collection:str = "rag_knowledge_database",
                namespace:str = "default",
                qdrant_url: str = None,
                qdrant_api_key: str = None,              
                llm: Optional[CoreLLM] = None          
                ):
        super().__init__(
            name="rag",
            description="RAG工具 - 支持多格式文档检索增强生成，提供智能问答能力"
        )
        self.knowledgeBaseURL = knowledgeBaseURL
        self.collection = collection
        self.namespace = namespace
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        
        # Initialize components
        self.llm = llm if llm is not None else CoreLLM()
        self.ingestion = RagIngestion()
        self.retriever = RagRetriever(self.llm)
        self._pipeline: Dict[str, Any] = {}
        
        # Ensure knowledge base directory exists
        os.makedirs(knowledgeBaseURL, exist_ok=True)
        self._initComponents()

    def _initComponents(self):
        """
        Initialize RAG components
        """
        try:
            # Initialize default namespace RAG pipeline
            default_pipeline = self.create_rag_pipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection=self.collection,
                namespace=self.namespace
            )
            self._pipeline[self.namespace] = default_pipeline
            self.initialized = True
            print(f"RAG tool initialized: namespace={self.namespace}, collection={self.collection}")
        except Exception as e:
            self.initialized = False
            print(f"RAG tool initialization failed: {e}")

    def create_rag_pipeline(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,     
        collection:str = "rag_knowledge_database",
        namespace:str = "default"
    ) -> Dict[str,Any]:
        """
        Create a complete RAG pipeline with Qdrant and unified embedding.
        """
        dimension = get_dimension(384)

        qdrantDB = QdrantVectorStore(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection,
            vector_size=dimension,
            distance="cosine"
        )

        # Add Document
        def add_documents(file_paths:List[str], chunk_size: int = 800, chunk_overlap:int = 100):
            """Add documents to RAG pipeline"""
            # Delegate loading and chunking to Ingestion
            chunks = self.ingestion.load_and_chunk_texts(
                paths=file_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                namespace=namespace,
                source_label="rag"
            )
            # Index chunk and store in index db (Rag responsibility)
            self.ingestion.index_chunks(
                store=qdrantDB,
                chunks=chunks,
                rag_namespace=namespace     
            )
            return len(chunks)
        
        # Search Similar Document
        def search(query:str, topk:int = 8, score_threshold: Optional[float]=None):
            """Search RAG knowledge base"""
            return self.retriever.search_vectors(
                store=qdrantDB,
                query=query,
                top_k=topk,
                namespace=namespace,
                score_threshold=score_threshold
            )
        
        # Advanced Search Similar Document
        def search_advanced(
            query: str, 
            top_k: int = 8, 
            enable_mqe: bool = False,
            enable_hyde: bool = False,
            score_threshold: Optional[float] = None
        ):
            """Advanced search with query expansion"""
            return self.retriever.search_vectors_expanded(
                store=qdrantDB,
                query=query,
                top_k=top_k,
                rag_namespace=namespace,
                enable_mqe=enable_mqe,
                enable_hyde=enable_hyde,
                score_threshold=score_threshold
            )
    
        def get_stats():
            """Get qdrantDB statistics"""
            return qdrantDB.get_collection_stats()
        
        return {
            "store": qdrantDB,
            "namespace": namespace,
            "add_documents": add_documents,
            "search": search,
            "search_advanced": search_advanced,
            "get_stats": get_stats
        }

    def run(self, parameters: Dict[str, Any]) -> str:
        """执行工具 - Tool基类要求的接口

        Args:
            parameters: 工具参数字典，必须包含action参数

        Returns:
            执行结果字符串
        """
        if not self.validate_parameters(parameters):
            return "❌ 参数验证失败：缺少必需的参数"

        action = parameters.get("action")
        # 移除action参数，传递其余参数给execute方法
        kwargs = {k: v for k, v in parameters.items() if k != "action"}

        return self.execute(action, **kwargs)

    def execute(self, action: str, **kwargs) -> str:
        """执行RAG操作
        
        主要操作流程：
        1. add_document/add_text: 数据 → 解析 → 分块 → 向量化 → 存储
        2. ask: 问题 → 检索 → 上下文注入 → LLM生成答案
        3. search: 查询 → 向量检索 → 返回相关片段
        """
        
        if not self.initialized:
            return f"❌ RAG工具未正确初始化，请检查配置: {getattr(self, 'init_error', '未知错误')}"
        
        # 参数预处理
        kwargs = self._preprocess_parameters(action, **kwargs)
        
        try:
            if action == "add_document":
                return self._add_document(**kwargs)
            elif action == "add_text":
                return self._add_text(**kwargs)
            elif action == "ask":
                return self._ask(**kwargs)
            elif action == "search":
                return self._search(**kwargs)
            elif action == "stats":
                return self._get_stats(namespace=kwargs.get("namespace"))
            elif action == "clear":
                return self._clear_knowledge_base(**kwargs)
            else:
                available_actions = ["add_document", "add_text", "ask", "search", "stats", "clear"]
                return f"❌ 不支持的操作: {action}\n✅ 可用操作: {', '.join(available_actions)}"
                
        except Exception as e:
            return f"❌ 执行操作 '{action}' 时发生错误: {str(e)}"
    
    def _add_document(self, 
        file_path: str, document_id: str = None, namespace: Optional[str] = None, 
        chunk_size: int = 800, chunk_overlap: int = 100, **kwargs) -> str:
        """添加文档到知识库（支持多格式）"""
        try:
            if not file_path or not os.path.exists(file_path):
                return f"❌ 文件不存在: {file_path}"
            
            pipeline = self._get_pipeline(namespace)
            t0 = time.time()

            chunks_added = pipeline["add_documents"](
                file_paths=[file_path],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            if chunks_added == 0:
                return f"⚠️ 未能从文件解析内容: {os.path.basename(file_path)}"
            
            return (
                f"✅ 文档已添加到知识库: {os.path.basename(file_path)}\n"
                f"📊 分块数量: {chunks_added}\n"
                f"⏱️ 处理时间: {process_ms}ms\n"
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}"
            )
            
        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"

    def _get_stats(self, namespace: Optional[str] = None) -> str:
        """获取知识库统计"""
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]()
            
            stats_info = [
                "📊 **RAG 知识库统计**",
                f"📝 命名空间: {pipeline.get('namespace', self.rag_namespace)}",
                f"📋 集合名称: {self.collection_name}",
                f"📂 存储根路径: {self.knowledge_base_path}"
            ]
            
            # 添加存储统计
            if stats:
                store_type = stats.get("store_type", "unknown")
                total_vectors = (
                    stats.get("points_count") or 
                    stats.get("vectors_count") or 
                    stats.get("count") or 0
                )
                
                stats_info.extend([
                    f"📦 存储类型: {store_type}",
                    f"📊 文档分块数: {int(total_vectors)}",
                ])
                
                if "config" in stats:
                    config = stats["config"]
                    if isinstance(config, dict):
                        vector_size = config.get("vector_size", "unknown")
                        distance = config.get("distance", "unknown")
                        stats_info.extend([
                            f"🔢 向量维度: {vector_size}",
                            f"📎 距离度量: {distance}"
                        ])
            
            # 添加系统状态
            stats_info.extend([
                "",
                "🟢 **系统状态**",
                f"✅ RAG 管道: {'正常' if self.initialized else '异常'}",
                f"✅ LLM 连接: {'正常' if hasattr(self, 'llm') else '异常'}"
            ])
            
            return "\n".join(stats_info)
            
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"
    
    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """获取指定命名空间的 RAG 管道，若不存在则自动创建"""
        target_ns = namespace or self.rag_namespace
        if target_ns in self._pipelines:
            return self._pipelines[target_ns]

        pipeline = self.create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection_name=self.collection_name,
            rag_namespace=target_ns
        )
        self._pipelines[target_ns] = pipeline
        return pipeline