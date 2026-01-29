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

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        return "action" in parameters

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
                f"📝 命名空间: {pipeline.get('namespace', self.namespace)}"
            )
            
        except Exception as e:
            return f"❌ 添加文档失败: {str(e)}"

    def _add_text(self, text: str, document_id: str = None, metadata: Optional[Dict[str, Any]] = None, namespace: Optional[str] = None, chunk_size: int = 800, chunk_overlap: int = 100, **kwargs) -> str:
        """添加文本到知识库"""
        try:
            if not text or not text.strip():
                return "❌ 文本内容不能为空"
            
            # 创建临时文件
            document_id = document_id or f"text_{abs(hash(text)) % 100000}"
            tmp_path = os.path.join(self.knowledgeBaseURL, f"{document_id}.md")
            
            try:
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                pipeline = self._get_pipeline(namespace)
                t0 = time.time()

                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                t1 = time.time()
                process_ms = int((t1 - t0) * 1000)
                
                if chunks_added == 0:
                    return f"⚠️ 未能从文本生成有效分块"
                
                return (
                    f"✅ 文本已添加到知识库: {document_id}\n"
                    f"📊 分块数量: {chunks_added}\n"
                    f"⏱️ 处理时间: {process_ms}ms\n"
                    f"📝 命名空间: {pipeline.get('namespace', self.namespace)}"
                )
                
            finally:
                # 清理临时文件
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
            
        except Exception as e:
            return f"❌ 添加文本失败: {str(e)}"
    
    def _search(self, query: str, limit: int = 5, min_score: float = 0.1, enable_advanced_search: bool = True, max_chars: int = 1200, include_citations: bool = True, namespace: Optional[str] = None, **kwargs) -> str:
        """搜索知识库"""
        try:
            if not query or not query.strip():
                return "❌ 搜索查询不能为空"
            
            # 使用统一 RAG 管道搜索
            pipeline = self._get_pipeline(namespace)

            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=query,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True,
                    score_threshold=min_score if min_score > 0 else None
                )
            else:
                results = pipeline["search"](
                    query=query,
                    top_k=limit,
                    score_threshold=min_score if min_score > 0 else None
                )
            
            if not results:
                return f"🔍 未找到与 '{query}' 相关的内容"
            
            # 格式化搜索结果
            search_result = ["搜索结果："]
            for i, result in enumerate(results, 1):
                meta = result.get("metadata", {})
                score = result.get("score", 0.0)
                content = meta.get("content", "")[:200] + "..."
                source = meta.get("source_path", "unknown")
                
                # 安全处理Unicode
                def clean_text(text):
                    try:
                        return str(text).encode('utf-8', errors='ignore').decode('utf-8')
                    except Exception:
                        return str(text)
                
                clean_content = clean_text(content)
                clean_source = clean_text(source)
                
                search_result.append(f"\n{i}. 文档: **{clean_source}** (相似度: {score:.3f})")
                search_result.append(f"   {clean_content}")
                
                if include_citations and meta.get("heading_path"):
                    clean_heading = clean_text(str(meta['heading_path']))
                    search_result.append(f"   章节: {clean_heading}")
            
            return "\n".join(search_result)
            
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"

    def _ask(self, question: Optional[str] = None, query: Optional[str] = None, limit: int = 5, enable_advanced_search: bool = True, include_citations: bool = True, max_chars: int = 1200, namespace: Optional[str] = None, **kwargs) -> str:
        """智能问答：检索 → 上下文注入 → LLM生成答案
        
        核心流程：
        1. 解析用户问题
        2. 智能检索相关内容
        3. 构建上下文和提示词
        4. LLM生成准确答案
        5. 添加引用来源
        """
        try:
            # 获取用户问题（question 优先级高于 query）
            user_question = question or query
            if not user_question or not user_question.strip():
                return "❌ 请提供要询问的问题"
            
            user_question = user_question.strip()
            print(f"🔍 智能问答: {user_question}")
            
            # 1. 检索相关内容
            pipeline = self._get_pipeline(namespace)
            search_start = time.time()
            
            if enable_advanced_search:
                results = pipeline["search_advanced"](
                    query=user_question,
                    top_k=limit,
                    enable_mqe=True,
                    enable_hyde=True
                )
            else:
                results = pipeline["search"](
                    query=user_question,
                    top_k=limit
                )
            
            search_time = int((time.time() - search_start) * 1000)
            
            if not results:
                return (
                    f"🤔 抱歉，我在知识库中没有找到与「{user_question}」相关的信息。\n\n"
                    f"💡 建议：\n"
                    f"• 尝试使用更简洁的关键词\n"
                    f"• 检查是否已添加相关文档\n"
                    f"• 使用 stats 操作查看知识库状态"
                )
            
            # 2. 智能整理上下文
            context_parts = []
            citations = []
            total_score = 0
            
            for i, result in enumerate(results):
                meta = result.get("metadata", {})
                content = meta.get("content", "").strip()
                source = meta.get("source_path", "unknown")
                score = result.get("score", 0.0)
                total_score += score
                
                if content:
                    # 清理内容格式
                    cleaned_content = self._clean_content_for_context(content)
                    context_parts.append(f"片段 {i+1}：{cleaned_content}")
                    
                    if include_citations:
                        citations.append({
                            "index": i+1,
                            "source": os.path.basename(source),
                            "score": score
                        })
            
            # 3. 构建上下文（智能截断）
            context = "\n\n".join(context_parts)
            if len(context) > max_chars:
                # 智能截断，保持完整性
                context = self._smart_truncate_context(context, max_chars)
            
            # 4. 构建增强提示词
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(user_question, context)
            
            enhanced_prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 5. 调用 LLM 生成答案
            llm_start = time.time()
            answer = self.llm.think(enhanced_prompt)
            llm_time = int((time.time() - llm_start) * 1000)
            
            if not answer or not answer.strip():
                return "❌ LLM未能生成有效答案，请稍后重试"
            
            # 6. 构建最终回答
            final_answer = self._format_final_answer(
                question=user_question,
                answer=answer.strip(),
                citations=citations if include_citations else None,
                search_time=search_time,
                llm_time=llm_time,
                avg_score=total_score / len(results) if results else 0
            )
            
            return final_answer
            
        except Exception as e:
            return f"❌ 智能问答失败: {str(e)}\n💡 请检查知识库状态或稍后重试"

    def _get_stats(self, namespace: Optional[str] = None) -> str:
        """获取知识库统计"""
        try:
            pipeline = self._get_pipeline(namespace)
            stats = pipeline["get_stats"]()
            
            stats_info = [
                "📊 **RAG 知识库统计**",
                f"📝 命名空间: {pipeline.get('namespace', self.namespace)}",
                f"📋 集合名称: {self.collection}",
                f"📂 存储根路径: {self.knowledgeBaseURL}"
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
        target_ns = namespace or self.namespace
        if target_ns in self._pipeline:
            return self._pipeline[target_ns]

        pipeline = self.create_rag_pipeline(
            qdrant_url=self.qdrant_url,
            qdrant_api_key=self.qdrant_api_key,
            collection=self.collection,
            namespace=target_ns
        )
        self._pipeline[target_ns] = pipeline
        return pipeline

    def _format_final_answer(self, question: str, answer: str, citations: Optional[List[Dict]] = None, search_time: int = 0, llm_time: int = 0, avg_score: float = 0) -> str:
        """格式化最终答案"""
        result = [f"🤖 **智能问答结果**\n"]
        result.append(answer)
        
        if citations:
            result.append("\n\n📚 **参考来源**")
            for citation in citations:
                score_emoji = "🟢" if citation["score"] > 0.8 else "🟡" if citation["score"] > 0.6 else "🔵"
                result.append(f"{score_emoji} [{citation['index']}] {citation['source']} (相似度: {citation['score']:.3f})")
        
        # 添加性能信息（调试模式）
        result.append(f"\n⚡ 检索: {search_time}ms | 生成: {llm_time}ms | 平均相似度: {avg_score:.3f}")
        
        return "\n".join(result)
    
    def _smart_truncate_context(self, context: str, max_chars: int) -> str:
        """智能截断上下文，保持段落完整性"""
        if len(context) <= max_chars:
            return context
        
        # 寻找最近的段落分隔符
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")
        
        if last_break > max_chars * 0.7:  # 如果断点位置合理
            return truncated[:last_break] + "\n\n[...更多内容被截断]"
        else:
            return truncated[:max_chars-20] + "...[内容被截断]"
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return (
            "你是一个专业的知识助手，具备以下能力：\n"
            "1. 📖 精准理解：仔细理解用户问题的核心意图\n"
            "2. 🎯 可信回答：严格基于提供的上下文信息回答，不编造内容\n"
            "3. 🔍 信息整合：从多个片段中提取关键信息，形成完整答案\n"
            "4. 💡 清晰表达：用简洁明了的语言回答，适当使用结构化格式\n"
            "5. 🚫 诚实表达：如果上下文不足以回答问题，请坦诚说明\n\n"
            "回答格式要求：\n"
            "• 直接回答核心问题\n"
            "• 必要时使用要点或步骤\n"
            "• 引用关键原文时使用引号\n"
            "• 避免重复和冗余"
        )
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        """构建用户提示词"""
        return (
            f"请基于以下上下文信息回答问题：\n\n"
            f"【问题】{question}\n\n"
            f"【相关上下文】\n{context}\n\n"
            f"【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"
        )
    
    def _clear_knowledge_base(self, confirm: bool = False, namespace: Optional[str] = None, **kwargs) -> str:
        """清空知识库"""
        try:
            if not confirm:
                return (
                    "⚠️ 危险操作：清空知识库将删除所有数据！\n"
                    "请使用 confirm=true 参数确认执行。"
                )
            
            pipeline = self._get_pipeline(namespace)
            store = pipeline.get("store")
            namespace_id = pipeline.get("namespace", self.namespace)
            success = store.clear_collection() if store else False
            
            if success:
                # 重新初始化该命名空间
                self._pipeline[namespace_id] = self.create_rag_pipeline(
                    qdrant_url=self.qdrant_url,
                    qdrant_api_key=self.qdrant_api_key,
                    collection=self.collection,
                    namespace=namespace_id
                )
                return f"✅ 知识库已成功清空（命名空间：{namespace_id}）"
            else:
                return "❌ 清空知识库失败"
            
        except Exception as e:
            return f"❌ 清空知识库失败: {str(e)}"
    
    def _clear_all_namespaces(self) -> str:
        """清空当前工具管理的所有命名空间数据"""
        try:
            for ns, pipeline in self._pipeline.items():
                store = pipeline.get("store")
                if store:
                    store.clear_collection()
            self._pipeline.clear()
            # 重新初始化默认命名空间
            self._init_components()
            return "✅ 所有命名空间数据已清空并重新初始化"
        except Exception as e:
            return f"❌ 清空所有命名空间失败: {str(e)}"

    def _clean_content_for_context(self, content: str) -> str:
        """清理内容用于上下文"""
        # 移除过多的换行和空格
        content = " ".join(content.split())
        # 截断过长内容
        if len(content) > 300:
            content = content[:300] + "..."
        return content

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        """为查询获取相关上下文
        
        这个方法可以被Agent调用来获取相关的知识库上下文
        """
        try:
            if not query:
                return ""
            
            # 使用统一 RAG 管道搜索
            pipeline = self._get_pipeline(namespace)
            results = pipeline["search"](
                query=query,
                top_k=limit
            )
            
            if not results:
                return ""
            
            # 合并上下文
            context_parts = []
            for result in results:
                content = result.get("metadata", {}).get("content", "")
                if content:
                    context_parts.append(content)
            
            merged_context = "\n\n".join(context_parts)
            
            # 限制长度
            if len(merged_context) > max_chars:
                merged_context = merged_context[:max_chars] + "..."
            
            return merged_context
            
        except Exception as e:
            return f"获取上下文失败: {str(e)}"

    def batch_add_texts(self, texts: List[str], document_ids: Optional[List[str]] = None, chunk_size: int = 800, chunk_overlap: int = 100, namespace: Optional[str] = None) -> str:
        """批量添加文本"""
        try:
            if not texts:
                return "❌ 文本列表不能为空"
            
            if document_ids and len(document_ids) != len(texts):
                return "❌ 文本数量和文档ID数量不匹配"
            
            pipeline = self._get_pipeline(namespace)
            t0 = time.time()
            
            total_chunks = 0
            successful_files = []
            
            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue
                    
                doc_id = document_ids[i] if document_ids else f"batch_text_{i}"
                tmp_path = os.path.join(self.knowledge_base_path, f"{doc_id}.md")
                
                try:
                    with open(tmp_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    chunks_added = pipeline["add_documents"](
                        file_paths=[tmp_path],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    total_chunks += chunks_added
                    successful_files.append(doc_id)
                    
                finally:
                    # 清理临时文件
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
            
            t1 = time.time()
            process_ms = int((t1 - t0) * 1000)
            
            return (
                f"✅ 批量添加完成\n"
                f"📊 成功文件: {len(successful_files)}/{len(texts)}\n"
                f"📊 总分块数: {total_chunks}\n"
                f"⏱️ 处理时间: {process_ms}ms"
            )
            
        except Exception as e:
            return f"❌ 批量添加失败: {str(e)}"

    def describe_inputs(self) -> List[ToolParameter]:
        """获取工具参数定义 - Tool基类要求的接口"""
        return [
            # 核心操作参数
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), stats(统计), clear(清空)",
                required=True
            ),
            
            # 内容参数
            ToolParameter(
                name="file_path",
                type="string",
                description="文档文件路径（支持PDF、Word、Excel、PPT、图片、音频等多种格式）",
                required=False
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文本内容",
                required=False
            ),
            ToolParameter(
                name="question",
                type="string", 
                description="用户问题（用于智能问答）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词（用于基础搜索）",
                required=False
            ),
            
            # 可选配置参数
            ToolParameter(
                name="namespace",
                type="string",
                description="知识库命名空间（用于隔离不同项目，默认：default）",
                required=False,
                default="default"
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量（默认：5）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="include_citations",
                type="boolean",
                description="是否包含引用来源（默认：true）",
                required=False,
                default=True
            )
        ]
    