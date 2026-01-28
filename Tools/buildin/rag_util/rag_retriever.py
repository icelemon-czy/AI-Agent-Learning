import os
from typing import List, Dict, Optional, Any
from Memory.embedding import get_text_embedder, get_dimension
from Memory.Storage.qdrantDB import QdrantConnectionManager, QdrantVectorStore
from Core.llm import CoreLLM

class RagRetriever:
    def __init__(self, llm: Optional[CoreLLM] = None):
        """
        Initialize RagRetriever.
        Args:
            llm: Optional CoreLLM instance. If not provided, will attempt to create a new one.
        """
        try:
            self.llm = llm or CoreLLM()
        except Exception as e:
            print(f"[WARNING] Failed to initialize CoreLLM for RagRetriever: {e}")
            self.llm = None

    def _create_default_vector_store(self, dimension: int = 384, collection_name: str = "rag_vectors") -> Any:
        # Create default Qdrant vector store with RAG-optimized settings.
        # 使用连接管理器避免重复连接。
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        return QdrantConnectionManager.get_instance(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=collection_name,
            vector_size=dimension,
            distance="cosine"
        )

    def search_vectors(
        self,
        store = None, 
        query: str = "", 
        top_k: int = 8, 
        namespace: Optional[str] = None, 
        only_rag_data: bool = True, 
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Search RAG vectors using unified embedding and Qdrant.
        """
        if not query:
            return []
        
        # Get default store if not provided
        if store is None:
            store = self._create_default_vector_store()
        
        # Convert query into embedding
        queryEmbedding = self._embed_query(query)
        
        # Build filter for RAG data
        where = {"memory_type": "rag_chunk"}
        if only_rag_data:
            where["is_rag_data"] = True
            where["data_source"] = "rag_pipeline"
        if namespace:
            where["rag_namespace"] = namespace
        
        try:
            return store.search_similar(
                query_vector=queryEmbedding, 
                limit=top_k, 
                score_threshold=score_threshold, 
                where=where
            )
        except Exception as e:
            print(f"[WARNING] RAG search failed: {e}")
            return []
    
    def search_vectors_expanded(
        self,
        store = None,
        query: str = "",
        top_k: int = 8,
        namespace: Optional[str] = None,
        only_rag_data: bool = True,
        score_threshold: Optional[float] = None,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
        rag_namespace: Optional[str] = None, # Added param alias for consistency
    ) -> List[Dict]:
        """
        Search with query expansion using unified embedding and Qdrant.
        """
        if not query:
            return []
        
        # Handle param alias
        if namespace is None and rag_namespace is not None:
            namespace = rag_namespace

        # Get default store if not provided
        if store is None:
            store = self._create_default_vector_store()
        
        # expansions
        expansions: List[str] = [query]
        
        if enable_mqe and mqe_expansions > 0:
            expansions.extend(self._prompt_mqe(query, mqe_expansions))
        if enable_hyde:
            hyde_text = self._prompt_hyde(query)
            if hyde_text:
                expansions.append(hyde_text)

        # unique and trim
        uniq: List[str] = []
        for e in expansions:
            if e and e not in uniq:
                uniq.append(e)
        expansions = uniq[: max(1, len(uniq))]

        # distribute pool per expansion
        pool = max(top_k * candidate_pool_multiplier, 20)
        per = max(1, pool // max(1, len(expansions)))

        # Build filter for RAG data
        where = {"memory_type": "rag_chunk"}
        if only_rag_data:
            where["is_rag_data"] = True
            where["data_source"] = "rag_pipeline"
        if namespace:
            where["rag_namespace"] = namespace

        # collect hits across expansions
        agg: Dict[str, Dict] = {}
        for q in expansions:
            qv = self._embed_query(q)
            hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)
            for h in hits:
                mid = h.get("metadata", {}).get("memory_id", h.get("id"))
                s = float(h.get("score", 0.0))
                if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                    agg[mid] = h
        # return top by score
        merged = list(agg.values())
        merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return merged[:top_k]

    def _embed_query(self, query: str) -> List[float]:
        """
        Embed query using unified embedding (百炼 with fallback).
        """
        embedder = get_text_embedder()
        dimension = get_dimension(384)
        try:
            vec = embedder.encode(query)
            
            # Normalize to List[float]
            if hasattr(vec, "tolist"):
                vec = vec.tolist()
            
            # 处理嵌套列表情况
            if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
                vec = vec[0]  # Extract first vector if nested
            
            # 转换为float列表
            result = [float(x) for x in vec]
            
            # 检查维度
            if len(result) != dimension:
                print(f"[WARNING] Query向量维度异常: 期望{dimension}, 实际{len(result)}")
                # 用零向量填充或截断
                if len(result) < dimension:
                    result.extend([0.0] * (dimension - len(result)))
                else:
                    result = result[:dimension]
            
            return result
        except Exception as e:
            print(f"[WARNING] Query embedding failed: {e}")
            # Return zero vector as fallback
            return [0.0] * dimension

    def _prompt_mqe(self, query: str, n: int) -> List[str]:
        try:
            if not self.llm:
                return [query]

            prompt = [
                {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
                {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
            ]
            text = self.llm.think(prompt)
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            outs = [ln for ln in lines if ln]
            return outs[:n] or [query]
        except Exception:
            return [query]

    def _prompt_hyde(self, query: str) -> Optional[str]:
        try:
            if not self.llm:
                return None
            
            prompt = [
                {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
                {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
            ]
            return self.llm.think(prompt)
        except Exception:
            return None
