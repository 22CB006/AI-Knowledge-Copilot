"""
Document processing module for extracting text from PDFs and web content.

This module provides the DocumentProcessor class that handles text extraction
from various sources including PDF files and web URLs.
"""
import uuid
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import pdfplumber
from PyPDF2 import PdfReader

from .models import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentProcessorError(Exception):
    """Base exception for document processing errors."""
    pass


class PDFProcessingError(DocumentProcessorError):
    """Exception raised when PDF processing fails."""
    pass


class WebContentError(DocumentProcessorError):
    """Exception raised when web content extraction fails."""
    pass


class DocumentProcessor:
    """
    Processes documents from various sources (PDF files, web URLs).
    
    This class handles text extraction from PDFs and web content,
    providing a unified interface for document ingestion.
    """
    
    def __init__(self):
        """Initialize the DocumentProcessor."""
        self.timeout = 30  # Timeout for web requests in seconds
        self.max_file_size = 50 * 1024 * 1024  # 50MB max file size
    
    def process_pdf(self, file_path: str) -> Document:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document object containing extracted text and metadata
            
        Raises:
            PDFProcessingError: If PDF processing fails
            FileNotFoundError: If the file doesn't exist
            
        Validates: Requirements 1.1, 1.3
        """
        try:
            path = Path(file_path)
            
            # Check if file exists
            if not path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            # Check file size
            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                raise PDFProcessingError(
                    f"PDF file too large: {file_size} bytes (max: {self.max_file_size})"
                )
            
            # Check if it's actually a PDF
            if not file_path.lower().endswith('.pdf'):
                raise PDFProcessingError(f"File is not a PDF: {file_path}")
            
            logger.info(f"Processing PDF: {file_path}")
            
            # Try pdfplumber first (better text extraction)
            text = self._extract_with_pdfplumber(file_path)
            
            # If pdfplumber fails or returns empty text, try PyPDF2
            if not text or not text.strip():
                logger.warning("pdfplumber returned empty text, trying PyPDF2")
                text = self._extract_with_pypdf2(file_path)
            
            # Validate that we got some text
            if not text or not text.strip():
                raise PDFProcessingError(
                    f"No text content extracted from PDF: {file_path}"
                )
            
            # Create document
            doc_id = str(uuid.uuid4())
            metadata = {
                "filename": path.name,
                "file_size": file_size,
                "file_path": str(path.absolute())
            }
            
            document = Document(
                id=doc_id,
                source=str(path.absolute()),
                source_type="pdf",
                text=text,
                metadata=metadata,
                created_at=datetime.now()
            )
            
            logger.info(f"Successfully processed PDF: {file_path} ({len(text)} characters)")
            return document
            
        except FileNotFoundError:
            raise
        except PDFProcessingError:
            raise
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise PDFProcessingError(f"Failed to process PDF: {str(e)}") from e
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """
        Extract text using pdfplumber.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {str(e)}")
            return ""
    
    def _extract_with_pypdf2(self, file_path: str) -> str:
        """
        Extract text using PyPDF2 as fallback.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            text_parts = []
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
            
            return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {str(e)}")
            return ""
    
    def process_url(self, url: str) -> Document:
        """
        Fetch and extract text from a web URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Document object containing extracted text and metadata
            
        Raises:
            WebContentError: If web content extraction fails
            
        Validates: Requirements 7.1
        """
        try:
            # Validate URL format
            if not url.startswith(('http://', 'https://')):
                raise WebContentError(f"Invalid URL format: {url}")
            
            logger.info(f"Fetching web content from: {url}")
            
            # Fetch the web page
            response = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            )
            
            # Check response status
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type.lower():
                raise WebContentError(
                    f"URL does not return HTML content: {content_type}"
                )
            
            # Extract text from HTML
            text = self._extract_text_from_html(response.text, url)
            
            # Validate that we got some text
            if not text or not text.strip():
                raise WebContentError(f"No text content extracted from URL: {url}")
            
            # Create document
            doc_id = str(uuid.uuid4())
            metadata = {
                "url": url,
                "content_type": content_type,
                "status_code": response.status_code
            }
            
            document = Document(
                id=doc_id,
                source=url,
                source_type="web",
                text=text,
                metadata=metadata,
                created_at=datetime.now()
            )
            
            logger.info(f"Successfully processed URL: {url} ({len(text)} characters)")
            return document
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout fetching URL: {url}")
            raise WebContentError(f"Request timeout for URL: {url}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error for URL: {url}")
            raise WebContentError(f"Failed to connect to URL: {url}")
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for URL {url}: {str(e)}")
            raise WebContentError(f"HTTP error: {str(e)}")
        except WebContentError:
            raise
        except Exception as e:
            logger.error(f"Error processing URL {url}: {str(e)}")
            raise WebContentError(f"Failed to process URL: {str(e)}") from e
    
    def _extract_text_from_html(self, html_content: str, url: str) -> str:
        """
        Extract text from HTML content using BeautifulSoup.
        
        Args:
            html_content: HTML content as string
            url: Source URL (for logging)
            
        Returns:
            Extracted text content
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from HTML for {url}: {str(e)}")
            raise WebContentError(f"Failed to parse HTML: {str(e)}") from e
    
    def extract_text(self, source: str, source_type: Optional[str] = None) -> str:
        """
        Extract text from a source (PDF file or URL).
        
        This is a convenience method that automatically determines the source type
        and calls the appropriate extraction method.
        
        Args:
            source: File path or URL
            source_type: Optional source type hint ("pdf" or "web")
            
        Returns:
            Extracted text content
            
        Raises:
            DocumentProcessorError: If extraction fails
        """
        try:
            # Auto-detect source type if not provided
            if source_type is None:
                if source.startswith(('http://', 'https://')):
                    source_type = 'web'
                elif source.lower().endswith('.pdf'):
                    source_type = 'pdf'
                else:
                    raise DocumentProcessorError(
                        f"Cannot determine source type for: {source}"
                    )
            
            # Process based on source type
            if source_type == 'pdf':
                document = self.process_pdf(source)
            elif source_type == 'web':
                document = self.process_url(source)
            else:
                raise DocumentProcessorError(
                    f"Unsupported source type: {source_type}"
                )
            
            return document.text
            
        except (PDFProcessingError, WebContentError):
            raise
        except Exception as e:
            logger.error(f"Error extracting text from {source}: {str(e)}")
            raise DocumentProcessorError(f"Failed to extract text: {str(e)}") from e
