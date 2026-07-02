# rag_pipeline/ingest.py
# Document ingestion: PDF → chunks with contextual metadata enrichment.
# Each chunk is prepended with source context to improve retrieval accuracy.

import os
from pathlib import Path

import pymupdf  # PyMuPDF (fitz)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import get_settings
from common.logger import get_logger

logger = get_logger(__name__)


def process_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text from a PDF page-by-page, splits into chunks,
    and enriches each chunk with contextual metadata.

    Returns a list of dicts: {'text': str, 'metadata': dict}
    """
    settings = get_settings()
    filename = os.path.basename(pdf_path)
    logger.info("Processing PDF: %s", filename)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    try:
        doc = pymupdf.open(pdf_path)
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            if not text or not text.strip():
                continue

            # Split the page text into chunks
            page_chunks = splitter.split_text(text)

            for chunk_idx, chunk_text in enumerate(page_chunks):
                # Contextual enrichment: prepend source info to the chunk.
                # This helps the embedding model understand WHERE the chunk came from,
                # significantly improving retrieval relevance.
                enriched_text = (
                    f"[Source: {filename} | Page: {page_num}] {chunk_text}"
                )

                metadata = {
                    "source": filename,
                    "page": page_num,
                    "chunk_index": chunk_idx,
                    "type": "pdf",
                }

                chunks.append({"text": enriched_text, "metadata": metadata})

        doc.close()
        logger.info("Extracted %d chunks from %s", len(chunks), filename)

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
