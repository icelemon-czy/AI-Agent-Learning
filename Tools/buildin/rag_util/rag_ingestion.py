import os
import hashlib
import typing
from typing import List, Dict, Optional, Any
from Memory.embedding import get_text_embedder, get_dimension
import re
import time

class RagIngestion:
    """
    Handles document loading, parsing, and chunking for RAG.
    """
    def __init__(self):
        pass

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
            return

        # Embedding model
        embedder = get_text_embedder()
        dimension = get_dimension(384)

        if store is None:
            raise ValueError("Vector store must be provided for indexing.")
        
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
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]
                else:
                    if part_vecs and not isinstance(part_vecs[0], (list, tuple)) and hasattr(part_vecs[0], "__len__"):
                        normalized_vecs = []
                        for v in part_vecs:
                            if hasattr(v, "tolist"):
                                normalized_vecs.append(v.tolist())
                            else:
                                normalized_vecs.append(list(v))
                        part_vecs = normalized_vecs
                    elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                        if hasattr(part_vecs, "tolist"):
                            part_vecs = [part_vecs.tolist()]
                        else:
                            part_vecs = [list(part_vecs)]
                for v in part_vecs:
                    try:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        v_norm = [float(x) for x in v]
                        if len(v_norm) != dimension:
                            if len(v_norm) < dimension:
                                v_norm.extend([0.0] * (dimension - len(v_norm)))
                            else:
                                v_norm = v_norm[:dimension]
                        vecs.append(v_norm)
                    except Exception as e:
                        print(f"[WARNING] Vector conversion failed: {e}, using zero vector")
                        vecs.append([0.0] * dimension)
            except Exception as e:
                print(f"[WARNING] Batch {i} encoding failed: {e}")
                print(f"[RAG] Retrying batch {i} with smaller chunks...")
                
                success = False
                for j in range(0, len(part), 8):
                    small_part = part[j:j+8]
                    try:
                        time.sleep(1)
                        small_vecs = embedder.encode(small_part)
                        if isinstance(small_vecs, list) and small_vecs and not isinstance(small_vecs[0], list):
                            small_vecs = [small_vecs]
                        
                        for v in small_vecs:
                            if hasattr(v, "tolist"):
                                v = v.tolist()
                            try:
                                v_norm = [float(x) for x in v]
                                if len(v_norm) != dimension:
                                    if len(v_norm) < dimension:
                                        v_norm.extend([0.0] * (dimension - len(v_norm)))
                                    else:
                                        v_norm = v_norm[:dimension]
                                vecs.append(v_norm)
                                success = True
                            except Exception as e2:
                                vecs.append([0.0] * dimension)
                    except Exception as e2:
                        for _ in range(len(small_part)):
                            vecs.append([0.0] * dimension)
                if not success:
                    print(f"[ERROR] Batch {i} failed completely")
            print(f"Embedding progress: {min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")

        # Prepare metadata 
        metas:List[Dict] = []
        ids: List[str] =[]
        for chunk in chunks:
            meta = {
                "memory_id": chunk["id"],
                "user_id": "rag_user",
                "memory_type": "rag_chunk",
                "content": chunk["content"],
                "data_source": "rag_pipeline",
                "rag_namespace": rag_namespace,
                "is_rag_data": True,
            }
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
                            "format": "markdown",
                        },
                    })

            print("End load and chunk documents.")
            return chunks

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
            return self._pdf_to_markdown(path)
        
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

    def _pdf_to_markdown(self,path: str) -> str:
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

    def _preprocess_markdown_for_embedding(self, text: str) -> str:
        """
        Preprocess markdown text for better embedding quality.
        """
        # Remove markdown headers symbols but keep the text
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown links but keep the text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove markdown emphasis markers
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove markdown code blocks but keep content
        text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()

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
