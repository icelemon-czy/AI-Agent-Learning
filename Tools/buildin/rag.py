from ..tool import Tool
from ...Core.llm import CoreLLM
from typing import Optional,Dict,Any,List
from ...Memory.embedding import get_text_embedder, get_dimension
from ...Memory.Storage.qdrantDB import QdrantVectorStore
import os
import hashlib
"""RAG工具 - 检索增强生成

提供简洁易用的RAG能力：
- 🔄 数据流程：用户数据 → 文档解析 → 向量化存储 → 智能检索 → LLM增强问答
- 📚 多格式支持：PDF、Word、Excel、PPT、图片、音频、网页等
- 🧠 智能问答：自动检索相关内容，注入提示词，生成准确答案
- 🏷️ 知识库管理：支持多项目隔离，便于管理不同知识库

使用示例：
```python
# 1. 初始化RAG工具
rag = RAGTool()

# 2. 添加文档
rag.run({"action": "add_document", "file_path": "document.pdf"})

# 3. 智能问答
answer = rag.run({"action": "ask", "question": "什么是机器学习？"})
```
"""

class Rag(Tool):
    def __init__(self,
                knowledgeBaseURL:str = "./knowledge_database",
                # 类似于数据库中的表, 物理隔离
                collection:str = "rag_knowledge_database",
                # 类似于数据库中的字段，逻辑隔离 （区分不同数据库/不同项目...)
                namespace:str = "default",
                # 向量数据库配置
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
        self.llm = llm
        if llm is None:
            self.llm = CoreLLM()
        
        # 确保知识库目录存在
        os.makedirs(knowledgeBaseURL, exist_ok=True)
        self._initComponents()

    def _initComponents(self):
        """
        初始化RAG组件
        """
        try:
            # 初始化默认命名空间的 RAG 管道
            default_pipeline = self.create_rag_pipeline(
                qdrant_url=self.qdrant_url,
                qdrant_api_key=self.qdrant_api_key,
                collection=self.collection,
                namespace=self.namespace
            )
            self._pipeline[self.namespace] = default_pipeline

            self.initialized = True
            print(f"RAG工具初始化成功: namespace={self.namespace}, collection={self.collection}")
        except Exception as e:
            self.initialized = False
            self.init_error = str(e)
            print(f"RAG工具初始化失败: {e}")

    def create_rag_pipeline(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,     
        collection:str = "rag_knowledge_database",
        namespace:str = "default"
    ) -> Dict[str,Any]:
        """
        Create a complete RAG pipeline with Qdrant and unified embedding.
        
        Returns:
            Dict containing store, namespace, and helper functions
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
            # load file and chunk 
            chunks = self.load_and_chunk_texts(
                paths=file_paths,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                namespace=namespace,
                source_label="rag"
            )
            # index chunk and store in index db.
            self.index_chunks(
                store=qdrantDB,
                chunks=chunks,
                rag_namespace=namespace     
            )
            return len(chunks)
        
        # Search Similar Document
        def search(query:str,topk:int = 8,score_threshold: Optional[float]=None):
            """Search RAG knowledge base"""
            return self.search_vectors(
                store=qdrantDB,
                query=query,
                top_k=topk,
                namespace=namespace,
                score_threshold=score_threshold
            )
        
        # Advamced Search Similar Document
        def search_advanced(
            query: str, 
            top_k: int = 8, 
            enable_mqe: bool = False,
            enable_hyde: bool = False,
            score_threshold: Optional[float] = None
        ):
            """Advanced search with query expansion"""
            """
            原理：不仅仅使用原始查询，还使用了查询扩展 (Query Expansion) 技术。
            MQE (Multi-Query Expansion)：利用 LLM 生成多个与原问题语义相关或互补的不同问法（例如问“ML是什么”，扩展为“机器学习定义”、“ML核心概念”）。
            HyDE (Hypothetical Document Embeddings)：利用 LLM 生成一个“假设性答案”，然后用这个答案去库里搜（因为答案和文档的相似度通常比问题和文档的相似度更高）。
            流程：原始查询 + LLM扩展查询 -> 分别搜索 -> 合并结果 -> 去重排序。
            特点：检索精度（Recall）通常更高，能召回一些字面不匹配但语义相关的文档，但因为要调 LLM 做扩展，速度较慢，且消耗 Token。
            适用场景：用户问题比较模糊、简短，或者标准检索效果不够好时。
            """
            return self.search_vectors_expanded(
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
            "store":qdrantDB,
            "namespace":namespace,
            "add_documents":add_documents,
            "search":search,
            "search_advanced":search_advanced,
            "get_stats":get_stats
        }
    
    def load_and_chunk_texts(
            self,
            paths: List[str], 
            chunk_size: int = 800, 
            chunk_overlap: int = 100, 
            namespace: Optional[str] = None, 
            source_label: str = "rag") -> List[Dict]:
            """
            Document loader and chunker using MarkItDown.
            Converts all supported formats to markdown, then chunks intelligently.
            """
            print("Start Load and Chunk documents.")
            seen_hashes = set()
            chunks:List[Dict]= []
            for path in paths:
                if not os.path.exists(path):
                    print(f"[WARNING] File not found: {path}")
                    continue  
                
                # Get File Suffix 
                file_ext = (os.path.splitext(path)[1] or '').lower()

                # Extract text by converting to markdown using MarkItDown
                markdown_text = self._convert_to_markdown(path)
                if not markdown_text.strip():
                    print(f"[WARNING] No content extracted from: {path}")
                    continue
                
                # Detect language
                language = self._detect_lang(markdown_text)

                # Generate Document ID
                doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode('utf-8')).hexdigest()

                # Split Markdown Texts into paragraphs
                paragraphs = self._split_paragraphs_with_headings(markdown_text)

                # Split paragraphs into small chunks
                token_chunks = self._chunk_paragraphs(paragraphs=paragraphs,chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap))

                for token_chunk in token_chunks:
                    content = token_chunk["content"]
                    start = token_chunk.get("start", 0)
                    end = token_chunk.get("end", start + len(content)) 
                    norm = content.strip()
                    if not norm:
                        continue

                    content_hash = hashlib.md5(norm.encode('utf-8')).hexdigest()
                    if content_hash in seen_hashes:
                        continue
                    seen_hashes.add(content_hash)
                    chunk_id = hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode('utf-8')).hexdigest()
                    chunks.append({
                        "id": chunk_id,
                        "content": content,
                        "metadata": {
                            "source_path": path,
                            "file_ext": file_ext,
                            "doc_id": doc_id,
                            "lang": language,
                            "start": start,
                            "end": end,
                            "content_hash": content_hash,
                            "namespace": namespace or "default",
                            "source": source_label,
                            "external": True,
                            "heading_path": token_chunk.get("heading_path"),
                            "format": "markdown",  # Mark all content as markdown-processed
                        },
                    })

            print("End load and chunk documents.")
            return chunks

    def index_chunks(self,
        store = None, 
        chunks: List[Dict] = None, 
        cache_db: Optional[str] = None, 
        batch_size: int = 64,
        rag_namespace: str = "default"
        ):
        """
        Index markdown chunks with unified embedding and Qdrant storage.
        """
        if chunks is None or len(chunks) == 0:
            print(f"No Chunks to index")  

        # Embedding model
        embedder = get_text_embedder()
        dimension = get_dimension(384)

        # Create Vector DB if not provided
        if store is None:
            store = self._create_default_vector_store(dimension=dimension,collection_name="rag_vectors")
            print(f"Created default Qdrant store with dimension {dimension}")
        
        # Preprocess markdown texts for better embeddings
        processed_texts = []
        for chunk in chunks:
            raw_content = chunk["content"]
            processed_content = self._preprocess_markdown_for_embedding(raw_content)
            processed_texts.append(processed_content)  

        print(f"Embedding start: total_texts={len(processed_texts)} batch_size={batch_size}")
        
        # Batch encoding with unified embedder
        vecs: List[List[float]] = []
        for i in range(0, len(processed_texts), batch_size):
            part = processed_texts[i:i+batch_size]
            try:
                # Use unified embedder directly (handles caching internally)
                part_vecs = embedder.encode(part)
                # Normalize to List[List[float]]
                if not isinstance(part_vecs, list):
                    # 单个numpy数组转为列表中的列表
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]
                else:
                    # 检查是否是嵌套列表
                    if part_vecs and not isinstance(part_vecs[0], (list, tuple)) and hasattr(part_vecs[0], "__len__"):
                        # numpy数组列表 -> 转换每个数组
                        normalized_vecs = []
                        for v in part_vecs:
                            if hasattr(v, "tolist"):
                                normalized_vecs.append(v.tolist())
                            else:
                                normalized_vecs.append(list(v))
                        part_vecs = normalized_vecs
                    elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                        # 单个向量被误判为列表，实际应该包装成[[...]]
                        if hasattr(part_vecs, "tolist"):
                            part_vecs = [part_vecs.tolist()]
                        else:
                            part_vecs = [list(part_vecs)]
                for v in part_vecs:
                    try:
                        # 确保向量是float列表
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        v_norm = [float(x) for x in v]
                        if len(v_norm) != dimension:
                            print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                            # 用零向量填充或截断
                            if len(v_norm) < dimension:
                                v_norm.extend([0.0] * (dimension - len(v_norm)))
                            else:
                                v_norm = v_norm[:dimension]
                        vecs.append(v_norm)
                    except Exception as e:
                        print(f"[WARNING] 向量转换失败: {e}, 使用零向量")
                        vecs.append([0.0] * dimension)
            except Exception as e:
                print(f"[WARNING] Batch {i} encoding failed: {e}")
                print(f"[RAG] Retrying batch {i} with smaller chunks...")
                # 尝试重试：将批次分解为更小的块
                success = False
                for j in range(0, len(part), 8):  # 更小的批次
                    small_part = part[j:j+8]
                    try:
                        import time
                        time.sleep(2)  # 等待2秒避免频率限制
                        
                        small_vecs = embedder.encode(small_part)
                        # Normalize to List[List[float]]
                        if isinstance(small_vecs, list) and small_vecs and not isinstance(small_vecs[0], list):
                            small_vecs = [small_vecs]
                        
                        for v in small_vecs:
                            if hasattr(v, "tolist"):
                                v = v.tolist()
                            try:
                                v_norm = [float(x) for x in v]
                                if len(v_norm) != dimension:
                                    print(f"[WARNING] 向量维度异常: 期望{dimension}, 实际{len(v_norm)}")
                                    if len(v_norm) < dimension:
                                        v_norm.extend([0.0] * (dimension - len(v_norm)))
                                    else:
                                        v_norm = v_norm[:dimension]
                                vecs.append(v_norm)
                                success = True
                            except Exception as e2:
                                print(f"[WARNING] 小批次向量转换失败: {e2}")
                                vecs.append([0.0] * dimension)
                    except Exception as e2:
                        print(f"[WARNING] 小批次 {j//8} 仍然失败: {e2}")
                        # 为这个小批次创建零向量
                        for _ in range(len(small_part)):
                            vecs.append([0.0] * dimension)
                if not success:
                    print(f"[ERROR] 批次 {i} 完全失败，使用零向量")
            print(f"Embedding progress: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")

        # Prepare metadata with RAG tags
        metas:List[Dict] = []
        ids: List[str] =[]
        for chunk in chunks:
            meta = {
                "memory_id": chunk["id"],
                "user_id": "rag_user",
                "memory_type": "rag_chunk",
                "content": chunk["content"],  # Keep original markdown content
                "data_source": "rag_pipeline",  # RAG identification tag
                "rag_namespace": rag_namespace,
                "is_rag_data": True,  # Clear RAG data marker
            }
            # Merge chunk metadata
            meta.update(chunk.get("metadata", {}))
            metas.append(meta)
            ids.append(chunk["id"])
        
        print(f"Qdrant upsert start: n={len(vecs)}")
        success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
        if success:
            print(f"Qdrant upsert done: {len(vecs)} vectors indexed")
        else:
            print(f"Qdrant upsert failed")
            raise RuntimeError("Failed to index vectors to Qdrant")

    def _preprocess_markdown_for_embedding(self,text: str) -> str:
        """
        Preprocess markdown text for better embedding quality.
        Removes excessive markup while preserving semantic content.
        """
        import re
        
        # Remove markdown headers symbols but keep the text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown emphasis markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # inline code
        
        # Remove markdown code blocks but keep content
        text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

    def _convert_to_markdown(
            self,path: str) -> str:
        """
        Document reader using MarkItDown with enhanced PDF processing.
        Converts any supported file format to markdown text.
        """
        if not os.path.exists(path):
            return ""
        
        # 对PDF文件使用增强处理
        ext = (os.path.splitext(path)[1] or '').lower()
        if ext == '.pdf':
            return self.__pdf_to_markdown(path)
        
        # 其他格式使用原有MarkItDown
        md_instance = self._get_markitdown_instance()
        if md_instance is None:
            return self._fallback_text_reader(path)
        
        try:
            result = md_instance.convert(path)
            text = getattr(result, "text_content", None)
            if isinstance(text, str) and text.strip():
                return text
            return ""
        except Exception as e:
            print(f"[WARNING] MarkItDown failed for {path}: {e}")
            return self._fallback_text_reader(path)         

    def __pdf_to_markdown(self,path: str) -> str:
        """
        Enhanced PDF processing with post-processing cleanup.
        """        
        # 使用原有MarkItDown提取
        md_instance = self._get_markitdown_instance()
        if md_instance is None:
            return self._fallback_text_reader(path)
        
        try:
            result = md_instance.convert(path)
            raw_text = getattr(result, "text_content", None)
            if not raw_text or not raw_text.strip():
                return ""
            
            # 后处理：清理和重组文本
            cleaned_text = self._post_process_pdf_text(raw_text)
            print(f"PDF post-processing completed: {len(raw_text)} -> {len(cleaned_text)} chars")
            return cleaned_text
            
        except Exception as e:
            print(f"[WARNING] Enhanced PDF processing failed for {path}: {e}")
            return self._fallback_text_reader(path)

    def _post_process_pdf_text(self,text: str) -> str:
        """
        Post-process PDF text to improve quality.
        """
        import re
        
        # 1. 按行分割并清理
        lines = text.splitlines()
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # 移除单个字符的行（通常是噪音）
            if len(line) <= 2 and not line.isdigit():
                continue
                
            # 移除明显的页眉页脚噪音
            if re.match(r'^\d+$', line):  # 纯数字行（页码）
                continue
            if line.lower() in ['github', 'project', 'forks', 'stars', 'language']:
                continue
                
            cleaned_lines.append(line)
        
        # 2. 智能合并短行
        merged_lines = []
        i = 0
        
        while i < len(cleaned_lines):
            current_line = cleaned_lines[i]
            
            # 如果当前行很短，尝试与下一行合并
            if len(current_line) < 60 and i + 1 < len(cleaned_lines):
                next_line = cleaned_lines[i + 1]
                
                # 合并条件：都是内容，不是标题
                if (not current_line.endswith('：') and 
                    not current_line.endswith(':') and
                    not current_line.startswith('#') and
                    not next_line.startswith('#') and
                    len(next_line) < 120):
                    
                    merged_line = current_line + " " + next_line
                    merged_lines.append(merged_line)
                    i += 2  # 跳过下一行
                    continue
            
            merged_lines.append(current_line)
            i += 1
        
        # 3. 重新组织段落
        paragraphs = []
        current_paragraph = []
        
        for line in merged_lines:
            # 检查是否是新段落的开始
            if (line.startswith('#') or  # 标题
                line.endswith('：') or   # 中文冒号结尾
                line.endswith(':') or    # 英文冒号结尾
                len(line) > 150 or       # 长句通常是段落开始
                not current_paragraph):  # 第一行
                
                # 保存当前段落
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                paragraphs.append(line)
            else:
                current_paragraph.append(line)
        
        # 添加最后一个段落
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        return '\n\n'.join(paragraphs)
    
    def _get_markitdown_instance(self):
        """
        Get a configured MarkItDown instance for document conversion.
        """
        try:
            from markitdown import MarkItDown
            return MarkItDown()
        except ImportError:
            print("[WARNING] MarkItDown not available. Install with: pip install markitdown")
            return None
    
    def _fallback_text_reader(self,path: str) -> str:
        """
        Simple fallback reader for basic text files when MarkItDown is unavailable.
        """
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            try:
                with open(path, 'r', encoding='latin-1', errors='ignore') as f:
                    return f.read()
            except Exception:
                return ""
    
    def _detect_lang(self,sample: str) -> str:
        """
        Detect language
        """
        try:
            from langdetect import detect
            return detect(sample[:1000]) if sample else "unknown"
        except Exception:
            return "unknown"
    
    def _split_paragraphs_with_headings(self,text: str) -> List[Dict]:
        """
        这个函数解析 Markdown 格式的文本，将其分割成结构化的段落列表，同时保留标题层次结构和位置信息。
        """
        lines = text.splitlines()
        heading_stack: List[str] = []
        paragraphs: List[Dict] = []
        buf: List[str] = []
        char_pos = 0
        def flush_buf(end_pos: int):
            if not buf:
                return
            content = "\n".join(buf).strip()
            if not content:
                return
            paragraphs.append({
                "content": content,
                "heading_path": " > ".join(heading_stack) if heading_stack else None,
                "start": max(0, end_pos - len(content)),
                "end": end_pos,
            })
        for ln in lines:
            raw = ln
            if raw.strip().startswith("#"):
                # heading line
                flush_buf(char_pos)
                level = len(raw) - len(raw.lstrip('#'))
                title = raw.lstrip('#').strip()
                if level <= 0:
                    level = 1
                if level <= len(heading_stack):
                    heading_stack = heading_stack[:level-1]
                heading_stack.append(title)
                char_pos += len(raw) + 1
                continue
            # paragraph accumulation
            if raw.strip() == "":
                flush_buf(char_pos)
                buf = []
            else:
                buf.append(raw)
            char_pos += len(raw) + 1
        flush_buf(char_pos)
        if not paragraphs:
            paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
        return paragraphs
    
    def _chunk_paragraphs(self,paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
        """
        这个函数实现了一个基于 token 数量的智能分块算法，用于将段落列表分割成更小的块，同时保留重叠以维持上下文连贯性。
        """
        chunks: List[Dict] = []
        cur: List[Dict] = []
        cur_tokens = 0
        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]
            p_tokens = self._approx_token_len(p["content"]) or 1
            if cur_tokens + p_tokens <= chunk_tokens or not cur:
                cur.append(p)
                cur_tokens += p_tokens
                i += 1
            else:
                # emit current chunk
                content = "\n\n".join(x["content"] for x in cur)
                start = cur[0]["start"]
                end = cur[-1]["end"]
                heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
                chunks.append({
                    "content": content,
                    "start": start,
                    "end": end,
                    "heading_path": heading_path,
                })
                # build overlap by keeping tail tokens
                if overlap_tokens > 0 and cur:
                    kept: List[Dict] = []
                    kept_tokens = 0
                    for x in reversed(cur):
                        t = self._approx_token_len(x["content"]) or 1
                        if kept_tokens + t > overlap_tokens:
                            break
                        kept.append(x)
                        kept_tokens += t
                    cur = list(reversed(kept))
                    cur_tokens = kept_tokens
                else:
                    cur = []
                    cur_tokens = 0
        if cur:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })
        return chunks

    def _approx_token_len(self,text: str) -> int:
        # 近似估计：CJK字符按1 token，其他按空白分词
        cjk = sum(1 for ch in text if self._is_cjk(ch))
        non_cjk_tokens = len([t for t in text.split() if t])
        return cjk + non_cjk_tokens

    def _is_cjk(self,ch: str) -> bool:
        code = ord(ch)
        return (
            0x4E00 <= code <= 0x9FFF or
            0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2A6DF or
            0x2A700 <= code <= 0x2B73F or
            0x2B740 <= code <= 0x2B81F or
            0x2B820 <= code <= 0x2CEAF or
            0xF900 <= code <= 0xFAFF
        )
    
    def _create_default_vector_store(dimension:int = 384, collection_name:str = "rag_vectors") -> QdrantVectorStore:
        """
        Create default Qdrant vector store with RAG-optimized settings.
        使用连接管理器避免重复连接。
        """
        # Check for Qdrant configuration
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        # 使用连接管理器
        from ...Memory.Storage.qdrantDB import QdrantConnectionManager
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
        queryEmbedding = self.embed_query(query)
        
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
    
    def embed_query(self,query: str) -> List[float]:
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
    ) -> List[Dict]:
        """
        Search with query expansion using unified embedding and Qdrant.
        """
        if not query:
            return []
        
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
            qv = self.embed_query(q)
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

    def _prompt_mqe(self,query: str, n: int) -> List[str]:
        try:
            from ...Core import llm
            llm = llm.CoreLLM()
            prompt = [
                {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
                {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
            ]
            text = llm.invoke(prompt)
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            outs = [ln for ln in lines if ln]
            return outs[:n] or [query]
        except Exception:
            return [query]

    def _prompt_hyde(self,query: str) -> Optional[str]:
        try:
            from ...Core import llm
            llm = llm.CoreLLM()
            prompt = [
                {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
                {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
            ]
            return llm.invoke(prompt)
        except Exception:
            return None

