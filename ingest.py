import hashlib
import os
import sys
from pathlib import Path

# Auto-detect and add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pymupdf  # PyMuPDF (fitz)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)


def compute_chunk_hash(text: str) -> str:
    """Generates a deterministic MD5 hash for a chunk text."""
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def process_pdf(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    """
    Extracts text from a PDF page-by-page, splits into chunks,
    and enriches each chunk with contextual metadata and a deterministic hash.

    Returns a list of dicts: {'text': str, 'metadata': dict}
    """
    settings = get_settings()
    filename = os.path.basename(pdf_path)
    logger.info("Processing PDF: %s", filename)

    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        logger.warning("PDF is missing or empty: %s", pdf_path)
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    seen_hashes = set()

    try:
        doc = pymupdf.open(pdf_path)
        if len(doc) == 0:
            logger.warning("PDF %s has 0 pages. Skipping.", filename)
            doc.close()
            return []

        for page_num, page in enumerate(doc, start=1):
            if max_pages and page_num > max_pages:
                break

            text = page.get_text()
            if not text or not text.strip():
                continue

            # Split the page text into chunks
            page_chunks = splitter.split_text(text)

            for chunk_idx, chunk_text in enumerate(page_chunks):
                chunk_hash = compute_chunk_hash(chunk_text)
                if chunk_hash in seen_hashes:
                    logger.debug("Skipping duplicate chunk in %s (p.%d)", filename, page_num)
                    continue
                seen_hashes.add(chunk_hash)

                # Contextual enrichment: prepend source info to the chunk.
                enriched_text = (
                    f"[Source: {filename} | Page: {page_num}] {chunk_text}"
                )

                metadata = {
                    "source": filename,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "chunk_hash": chunk_hash,
                    "type": "pdf",
                }

                chunks.append({"text": enriched_text, "metadata": metadata})

        doc.close()
        logger.info("Extracted %d unique chunks from %s", len(chunks), filename)

    except Exception as e:
        logger.error("Failed to process PDF %s: %s", pdf_path, e)

    return chunks


def ingest_directory(directory_path: str | None = None) -> list[dict]:
    """
    Ingests all supported files from a directory.

    Returns a list of all document chunks with metadata.
    """
    settings = get_settings()
    dir_path = directory_path or settings.data_source_dir

    # Resolve relative paths from project root
    if not os.path.isabs(dir_path):
        from common.config import PROJECT_ROOT
        dir_path = str(PROJECT_ROOT / dir_path)

    if not os.path.exists(dir_path):
        logger.warning("Data directory does not exist: %s", dir_path)
        return []

    all_chunks = []
    supported_extensions = {".pdf": process_pdf}

    files = sorted(os.listdir(dir_path))
    logger.info("Scanning directory: %s (%d files found)", dir_path, len(files))

    for filename in files:
        file_path = os.path.join(dir_path, filename)
        ext = Path(filename).suffix.lower()

        if ext in supported_extensions:
            processor = supported_extensions[ext]
            chunks = processor(file_path)
            all_chunks.extend(chunks)
        else:
            logger.debug("Skipping unsupported file: %s", filename)

    logger.info("Total chunks ingested: %d", len(all_chunks))
    return all_chunks
