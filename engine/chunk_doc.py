#  IMPORT FRAMEWORK 
from langchain_experimental.text_splitter   import SemanticChunker
from langchain.text_splitter                import RecursiveCharacterTextSplitter
from langchain.docstore.document            import Document

from werkzeug.local import Local
from bs4 import BeautifulSoup
import re
import requests
import tiktoken
import fitz # PyMuPDF
from typing import Optional, List
from models.embeddings import EmbeddingsModels
from google.api_core.exceptions import ResourceExhausted 
import time

from helpers.astradb_connect_helper import get_vector_collection, get_astradb_doc_chunked

from setup import LOGGER

class TokenCounter:
    """
    Manages token counting for semantic chunking
    """
    _local = Local()
    _encoding = tiktoken.encoding_for_model("text-embedding-3-large")

    @classmethod
    def set_semantic_chunker_token(cls, token: int) -> None:
        cls._local.semantic_chunker_token = token

    @classmethod
    def get_semantic_chunker_token(cls) -> Optional[int]:
        return getattr(cls._local, "semantic_chunker_token", None)

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Count tokens in a text string"""
        return len(cls._encoding.encode(text))
    
    @classmethod
    def tokens_embedding(cls, text: str) -> int:
        """Alias for count_tokens, used for embedding token counts."""
        return cls.count_tokens(text)

    @classmethod
    def tokens_semantic_chunker(cls, text: str) -> int:
        """Count tokens for the semantic chunker input."""
        return cls.count_tokens(text)
    
    
class DocProcess: 
    """
    End to nd PDF processing pipeline:
    1. Download PDF from URL.
    2. Parse HTML from each page and retain headings.
    3. Clean line breaks / whitespace.
    4. Semantic chunk the text with adaptive thresholding.
    5. Extract hierarchical headers (h1, h2, h3) into metadata.
    6. Re split oversize chunks, enrich with course metadata, token count quota check.
    7. Embed chunks and store in vector database.

    Args:
        threshold (int): Initial percentile breakpoint for semantic chunking.
        max_iterations (int): Max binary search iterations when re splitting long chunks.
        max_char_length (int): Maximum characters allowed in a single chunk before re split.
        embed_quota (int): Token quota for one document ingestion.
    """
    def __init__(self, 
                 threshold: int = 70, 
                 max_iterations: int = 2,
                 max_char_length: int = 7000,
                 embed_quota= 1_000_000,
                 ) -> None:
        self.threshold = threshold 
        self.max_iterations = max_iterations 
        self.max_iterations = max_iterations
        self.embed_quota = embed_quota
        self.max_char_length = max_char_length
        
    def load_pdf_url(self, pdf_url: str) -> fitz.Document:
        """
        Download a PDF from a URL and open it with PyMuPDF.

        Args:
            pdf_url (str): URL expected to serve a PDF.

        Returns:
            fitz.Document: Open PDF document object.

        Raises:
            ValueError: If URL is invalid or not a PDF or download/open fails.
        """
        if not isinstance(pdf_url, str):
            LOGGER.error("pdf_url must be a string")
            raise ValueError("Invalid pdf_url: must be a string")

        try:
            # ***** HTTP GET the PDF *****
            response = requests.get(pdf_url)
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "").lower()
            if "application/pdf" not in content_type:
                LOGGER.error("URL did not return a PDF. Content-Type: %s", content_type)
                raise ValueError(f"URL did not return a PDF. Content-Type: {content_type}")

            return fitz.open(stream=response.content, filetype="pdf")

        except Exception as error:
            LOGGER.error("Failed to download PDF: %s", error)
            raise ValueError(f"Failed to load PDF: {error}") from error
    
    def tokens_semantic_chunker(self, data: str, buffer_size: int = 1) -> int:
        """
        Count tokens for sliding window sentence contexts.

        Args:
            data (str): Full document text.
            buffer_size (int): # sentences to include before/after each sentence.

        Returns:
            int: Aggregate token count across all windows.
        """
        # ***** Split text into sentences *****
        single_sentences = re.split(r'(?<=[.?!])\s+', data)
        n = len(single_sentences)

        # ***** Count tokens per sentence once *****
        sentence_tokens: List[int] = [
            TokenCounter.count_tokens(sent) for sent in single_sentences
        ]

        total_tokens = 0
        for i in range(n):
            start = max(0, i - buffer_size)
            end = min(n, i + buffer_size + 1)
            total_tokens += sum(sentence_tokens[j] for j in range(start, end))

        LOGGER.info(
            "Done counting tokens for semantic chunking. "
            "Buffer size=%d, sentences=%d, total tokens=%d",
            buffer_size, n, total_tokens
        )
        return total_tokens
    
    def parsing_pdf_html(self, pdf_object: fitz.Document) -> str:
        """
        Extract headings and body text from each page's XHTML.

        Args:
            pdf_object (fitz.Document): Open PDF.

        Returns:
            str: Concatenated HTML‑like string with headings preserved.
        """
        join_page = ''
        for page in pdf_object: 
            page_html = page.get_textpage().extractXHTML() 
            search_html = BeautifulSoup(page_html, "html.parser")
            
            for line in search_html.div:
                # ***** Preserve headings h1‑h4, ignore italics *****
                if line.name in {"h1", "h2", "h3"} and not line.find("i"):
                    join_page += f'{line}'
                else: 
                    join_page += f'{line.text} '
        
        LOGGER.info('Parsing the document is Done')
        return join_page.strip()
    
    def remove_break_add_whitespace(self, text: str) -> str:
        """
        Replace single line breaks with spaces and double line breaks with paragraph gaps.

        Args:
            text (str): Raw extracted text.

        Returns:
            str: Cleaned text.
        """
        pattern = r"(\n)([a-z])"
        pattern2 = r"\n"
        replace_break = re.sub(pattern, r" \2", text)
        replace_break = re.sub(pattern2, r"\n\n", replace_break)
        LOGGER.info("Whitespace cleaned")
        return replace_break
    
    def create_document_by_splitting(self, page: str, threshold_amount: int = None) -> list[Document]:
        """
        Semantic chunk the cleaned page content into LangChain Documents.

        Args:
            page (str): Clean HTML/text string.
            threshold_amount (int, optional): Overrides default threshold.

        Returns:
            List[Document]: List of semantically coherent chunks.
        """
        if not isinstance(page, str):
            LOGGER.error("page for doc splitting must be a string")
            raise ValueError("page must be str")
        
        # ***** Precompute tokens for semantic chunker normalization *****
        tokens_for_semantic_chunker = self.tokens_semantic_chunker(page)
        TokenCounter.set_semantic_chunker_token(tokens_for_semantic_chunker)
       
        try:
            LOGGER.info("Chunking Procces...")
            text_splitter = SemanticChunker(
                EmbeddingsModels().embedding_large_gemini, 
                breakpoint_threshold_type ="percentile", 
                breakpoint_threshold_amount = threshold_amount if threshold_amount else self.threshold
                )
            docs = text_splitter.create_documents(texts=[page])
            LOGGER.info(f"Chunking is done... {len(docs)}")
            return docs
        except Exception as error: 
            raise ValueError(f'Create Document by splitting failed {error}')
        
    def extract_headers(self, document: List[Document]) -> List[Document]:
        """
        Extract h1–h3 hierarchy for each chunk and store in metadata.

        Args:
            document (List[Document]): List of chunks with raw HTML.

        Returns:
            List[Document]: Same chunks with 'header1/2/3' metadata added.
        """
        try:
            clean = re.compile('<.*?>')
            docs = []
            h1 = h2 = h3  = None

            for chunk in document:
                # ***** Strip HTML tags for plain text *****
                clean_content = re.sub(clean, '', chunk.page_content)
                temp_doc = Document(page_content=clean_content)
                
                # ***** Parse headings from original HTML *****
                soup = BeautifulSoup(chunk.page_content, "html.parser")
                for line in soup.find_all(['h1', 'h2', 'h3']):
                    text = line.get_text(strip=True)
                    if line.name == 'h1':
                        h1, h2, h3 = text, None, None
                    elif line.name == 'h2':
                        h2, h3 = text, None
                    elif line.name == 'h3':
                        h3 = text
                temp_doc.metadata = {'header1': h1, 'header2': h2, 'header3': h3}
                docs.append(temp_doc)
                
            LOGGER.info("Header extraction done")
            return docs
        
        except Exception as error:
            LOGGER.error(f"Header extraction failed: {str(error)}")
            raise ValueError(f"Failed to extract headers: {str(error)}")
    
    def _enrich_and_count(
        self,
        doc,
        course_id: str,
        course_name: str,
        doc_name: str,
        doc_id: str
    ) -> tuple[Document, int]:
        """
        Add metadata to a chunk and return token count for embedding.

        Returns:
            Tuple[Document, int]: Updated doc and token count.
        """
        tokens = TokenCounter.tokens_embedding(doc.page_content)
        metadata = {
            "course_id": course_id,
            "course_name": course_name,
            "document_name": doc_name,
            "document_id": doc_id,
            "tokens_embbed": tokens,
        }
        doc.metadata.update(metadata)
        return doc, tokens
    
    def resplit_chunks(self, chunked_doc) -> List[object]:
        """
        Binary search the percentile threshold to ensure chunks stay below max_char_length,
        else fall back to recursive character splitter.

        Args:
            chunked_doc (Document): A single oversized chunk.

        Returns:
            List[Document]: List of re‑split chunks.
        """
        low, high = 0, self.initial_threshold
        best_docs = None

        for _ in range(self.max_iterations):
            mid = (low + high) // 2
            docs = self.create_document_by_splitting(chunked_doc.page_content, mid)
            if all(len(d.page_content) <= self.max_char_length for d in docs):
                best_docs, high = docs, mid - 1
            else:
                low = mid + 1
            if low > high:
                break

        if best_docs:
            return best_docs

        LOGGER.warning(
            "Semantic splitting failed after %d iterations; using recursive splitter",
            self.max_iterations
        )
        # Fallback to recursive character splitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=80
        )
        return recursive_splitter.create_documents(texts=[chunked_doc.page_content])
    
    def embed_client_side(self, chunks: list[Document]) -> None:
        """
        Embed document chunks and store in vector collection.

        Args:
            chunks (List[Document]): Documents to embed.

        Returns:
            None
        """
        try:
            vector_coll = get_vector_collection()
            
            # ***** Sanity‑check embedding shape *****
            if chunks: 
                test_embed = vector_coll.embeddings.embed_query(chunks[0].page_content)
                print(f"Test embedding type: {type(test_embed)}, length: {len(test_embed)}")
            
            vector_coll.add_documents(chunks)
        except ResourceExhausted as error:
            LOGGER.error(f"Fail rate limit error. Call error webhook {error}")
            
    def process_documents(self,
                          url: str, 
                          course_id: str, 
                          course_name:str,  
                          doc_name: str, 
                          doc_id: str) -> None: 
        """
        Orchestrate the full PDF ingestion pipeline.

        Args:
            url (str): PDF URL.
            course_id (str): Course identifier.
            course_name (str): Course display name.
            doc_name (str): Friendly document name.
            doc_id (str): Unique document identifier.

        Returns:
            None
        """
        start_time = time.time()

        # ***** Load PDF *****
        pages = self.load_pdf_url(url)
        if not pages:
            raise ValueError("Failed to load PDF")
        LOGGER.info("PDF loaded: %s", doc_name)

        # ***** Parse & clean *****
        parsed = self.parsing_pdf_html(pages)
        cleaned = self.remove_break_add_whitespace(parsed)

        # ***** Initial chunking & header extraction *****
        split_docs = self.create_document_by_splitting(cleaned)
        header_docs = self.extract_headers(split_docs)

        # ***** Re‑split long chunks, enrich metadata, count tokens *****
        all_chunks: List[Document] = []
        total_tokens = 0

        for chunk in header_docs:
            docs = (
                self.resplit_chunks(chunk)
                if len(chunk.page_content) > self.max_char_length
                else [chunk]
            )
            for doc in docs:
                enriched, count = self._enrich_and_count(
                    doc, course_id, course_name, doc_name, doc_id
                )
                all_chunks.append(enriched)
                total_tokens += count

        total_tokens += TokenCounter.get_semantic_chunker_token()
        LOGGER.info("Total tokens: %d", total_tokens)
        if total_tokens > self.embed_quota:
            raise ValueError(f"Quota exceeded: {total_tokens} > {self.embed_quota}")

        # ***** Embed and store *****
        self.embed_client_side(all_chunks)

        end_time = time.time()
        LOGGER.info(
            "Finished embedding %d chunks | Total time: %.2fs",
            len(all_chunks), end_time - start_time
        )

    def _delete_document(self, doc_id: str) -> None: 
        """
        Delete all vector chunks associated with a document ID.

        Args:
            doc_id (str): Document identifier.

        Returns:
            None
        """
        collection = get_astradb_doc_chunked()
        collection.delete_many(filter={"metadata.document_id": doc_id})
        LOGGER.info('Deleted done')
    
if __name__ == '__main__': 
    doc_url = 'https://arxiv.org/pdf/1706.03762'
    process_doc = DocProcess()
    process_doc.process_documents(doc_url, "id", "name", "doc_name", 'doc_id')