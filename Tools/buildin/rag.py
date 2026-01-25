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
                knowledgeBaseURL:str = "./knowledge_base",
                knowledgeNameSpace:str = "default",
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
        self.knowledgeNameSpaces = [knowledgeNameSpace]
        self.currentKnowledgeNameSpace = knowledgeNameSpace
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.llm = llm
        
        # 确保知识库目录存在
        os.makedirs(knowledgeBaseURL, exist_ok=True)
    
    def create_rag_pipeline(
        self,
        qdrant_url: str = None,
        qdrant_api_key: str = None,     
        knowledgeBaseURL:str = "./knowledge_base",
        knowledgeNameSpace:str = "default"
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
            collection_name=knowledgeNameSpace,
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
                namespace=knowledgeNameSpace,
                source_label="rag"
            )
            # index chunk and store in index db.
            self.index_chunks()
        
        # Search Similar Document
        def search(query:str,topk:int = 8,score_threshold: Optional[float]=None):
            pass

        pass

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

    def index_chunks(self):
        pass

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