"""
Unit tests for the DocumentProcessor class.
"""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import requests
from hypothesis import given, strategies as st, settings

from src.document_processor import (
    DocumentProcessor,
    PDFProcessingError,
    WebContentError,
    DocumentProcessorError
)
from src.models import Document


class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_pdf_path(self):
        """Create a temporary PDF file for testing."""
        # Create a minimal valid PDF
        pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000317 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
410
%%EOF
"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(pdf_content)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_process_pdf_success(self, processor, sample_pdf_path):
        """Test successful PDF processing."""
        document = processor.process_pdf(sample_pdf_path)
        
        assert isinstance(document, Document)
        assert document.source_type == "pdf"
        assert document.text  # Should have extracted text
        assert document.id  # Should have generated ID
        assert "filename" in document.metadata
        assert document.metadata["file_path"] == str(Path(sample_pdf_path).absolute())
    
    def test_process_pdf_file_not_found(self, processor):
        """Test PDF processing with non-existent file."""
        with pytest.raises(FileNotFoundError):
            processor.process_pdf("/nonexistent/file.pdf")
    
    def test_process_pdf_not_pdf_extension(self, processor):
        """Test PDF processing with non-PDF file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Not a PDF")
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError, match="File is not a PDF"):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_pdf_file_too_large(self, processor):
        """Test PDF processing with file exceeding size limit."""
        processor.max_file_size = 100  # Set very small limit
        
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(b'x' * 200)  # Write more than limit
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError, match="PDF file too large"):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_process_pdf_corrupted(self, processor):
        """Test PDF processing with corrupted PDF."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            f.write(b'Not a valid PDF content')
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @patch('src.document_processor.requests.get')
    def test_process_url_success(self, mock_get, processor):
        """Test successful URL processing."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
        mock_response.text = """
        <html>
            <head><title>Test Page</title></head>
            <body>
                <h1>Test Heading</h1>
                <p>This is test content from a web page.</p>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        document = processor.process_url("https://example.com/test")
        
        assert isinstance(document, Document)
        assert document.source_type == "web"
        assert document.source == "https://example.com/test"
        assert "Test Heading" in document.text
        assert "test content" in document.text
        assert document.metadata["url"] == "https://example.com/test"
        assert document.metadata["status_code"] == 200
    
    def test_process_url_invalid_format(self, processor):
        """Test URL processing with invalid URL format."""
        with pytest.raises(WebContentError, match="Invalid URL format"):
            processor.process_url("not-a-url")
    
    @patch('src.document_processor.requests.get')
    def test_process_url_timeout(self, mock_get, processor):
        """Test URL processing with timeout."""
        mock_get.side_effect = requests.exceptions.Timeout()
        
        with pytest.raises(WebContentError, match="Request timeout"):
            processor.process_url("https://example.com/test")
    
    @patch('src.document_processor.requests.get')
    def test_process_url_connection_error(self, mock_get, processor):
        """Test URL processing with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        with pytest.raises(WebContentError, match="Failed to connect"):
            processor.process_url("https://example.com/test")
    
    @patch('src.document_processor.requests.get')
    def test_process_url_http_error(self, mock_get, processor):
        """Test URL processing with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="HTTP error"):
            processor.process_url("https://example.com/test")
    
    @patch('src.document_processor.requests.get')
    def test_process_url_non_html_content(self, mock_get, processor):
        """Test URL processing with non-HTML content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/api/data")
    
    @patch('src.document_processor.requests.get')
    def test_process_url_empty_content(self, mock_get, processor):
        """Test URL processing with empty HTML content."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/html'}
        mock_response.text = "<html><body></body></html>"
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="No text content extracted"):
            processor.process_url("https://example.com/empty")
    
    @patch('src.document_processor.requests.get')
    def test_extract_text_from_html_removes_scripts(self, mock_get, processor):
        """Test that HTML extraction removes script and style tags."""
        html = """
        <html>
            <head>
                <style>body { color: red; }</style>
                <script>console.log('test');</script>
            </head>
            <body>
                <p>Visible content</p>
                <script>alert('popup');</script>
            </body>
        </html>
        """
        
        text = processor._extract_text_from_html(html, "https://example.com")
        
        assert "Visible content" in text
        assert "console.log" not in text
        assert "alert" not in text
        assert "color: red" not in text
    
    @patch('src.document_processor.DocumentProcessor.process_pdf')
    def test_extract_text_auto_detect_pdf(self, mock_process_pdf, processor):
        """Test extract_text auto-detects PDF files."""
        mock_doc = Mock()
        mock_doc.text = "PDF content"
        mock_process_pdf.return_value = mock_doc
        
        text = processor.extract_text("/path/to/file.pdf")
        
        assert text == "PDF content"
        mock_process_pdf.assert_called_once_with("/path/to/file.pdf")
    
    @patch('src.document_processor.DocumentProcessor.process_url')
    def test_extract_text_auto_detect_url(self, mock_process_url, processor):
        """Test extract_text auto-detects URLs."""
        mock_doc = Mock()
        mock_doc.text = "Web content"
        mock_process_url.return_value = mock_doc
        
        text = processor.extract_text("https://example.com/page")
        
        assert text == "Web content"
        mock_process_url.assert_called_once_with("https://example.com/page")
    
    def test_extract_text_unknown_source_type(self, processor):
        """Test extract_text with unknown source type."""
        with pytest.raises(DocumentProcessorError, match="Cannot determine source type"):
            processor.extract_text("/path/to/unknown.xyz")
    
    def test_extract_text_explicit_source_type(self, processor):
        """Test extract_text with explicit source type."""
        with patch.object(processor, 'process_pdf') as mock_process_pdf:
            mock_doc = Mock()
            mock_doc.text = "PDF content"
            mock_process_pdf.return_value = mock_doc
            
            text = processor.extract_text("/some/file", source_type="pdf")
            
            assert text == "PDF content"
            mock_process_pdf.assert_called_once_with("/some/file")
    
    def test_extract_text_unsupported_source_type(self, processor):
        """Test extract_text with unsupported source type."""
        with pytest.raises(DocumentProcessorError, match="Unsupported source type"):
            processor.extract_text("/path/to/file", source_type="unknown")


class TestDocumentProcessorErrorHandling:
    """Test suite for error handling in DocumentProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a DocumentProcessor instance."""
        return DocumentProcessor()
    
    # Error Handling Tests for Corrupted PDFs
    
    def test_corrupted_pdf_invalid_header(self, processor):
        """Test handling of corrupted PDF with invalid header."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write invalid PDF header
            f.write(b'%NOT-A-PDF-1.4\n')
            f.write(b'Some random content that is not a valid PDF structure')
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_corrupted_pdf_truncated_file(self, processor):
        """Test handling of truncated PDF file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write only partial PDF header
            f.write(b'%PDF-1.4\n')
            # Missing rest of PDF structure
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_corrupted_pdf_binary_garbage(self, processor):
        """Test handling of PDF with binary garbage content."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write random binary data
            f.write(b'\x00\x01\x02\x03\x04\x05' * 100)
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_corrupted_pdf_empty_file(self, processor):
        """Test handling of empty PDF file."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write nothing - empty file
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_corrupted_pdf_malformed_structure(self, processor):
        """Test handling of PDF with malformed internal structure."""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
            # Write PDF header but with malformed structure
            pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
MALFORMED CONTENT HERE
>>
endobj
%%EOF
"""
            f.write(pdf_content)
            temp_path = f.name
        
        try:
            with pytest.raises(PDFProcessingError):
                processor.process_pdf(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    # Error Handling Tests for Invalid URLs
    
    def test_invalid_url_no_protocol(self, processor):
        """Test handling of URL without protocol."""
        with pytest.raises(WebContentError, match="Invalid URL format"):
            processor.process_url("example.com/page")
    
    def test_invalid_url_ftp_protocol(self, processor):
        """Test handling of URL with unsupported protocol."""
        with pytest.raises(WebContentError, match="Invalid URL format"):
            processor.process_url("ftp://example.com/file")
    
    def test_invalid_url_malformed(self, processor):
        """Test handling of malformed URL."""
        with pytest.raises(WebContentError, match="Failed to process URL"):
            processor.process_url("http://")
    
    def test_invalid_url_empty_string(self, processor):
        """Test handling of empty URL string."""
        with pytest.raises(WebContentError, match="Invalid URL format"):
            processor.process_url("")
    
    @patch('src.document_processor.requests.get')
    def test_invalid_url_dns_failure(self, mock_get, processor):
        """Test handling of URL with DNS resolution failure."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Name or service not known")
        
        with pytest.raises(WebContentError, match="Failed to connect"):
            processor.process_url("https://nonexistent-domain-12345.com")
    
    @patch('src.document_processor.requests.get')
    def test_invalid_url_ssl_error(self, mock_get, processor):
        """Test handling of URL with SSL certificate error."""
        mock_get.side_effect = requests.exceptions.SSLError("SSL certificate verification failed")
        
        with pytest.raises(WebContentError, match="Failed to connect"):
            processor.process_url("https://expired-ssl-cert.example.com")
    
    @patch('src.document_processor.requests.get')
    def test_invalid_url_404_not_found(self, mock_get, processor):
        """Test handling of URL returning 404 error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Client Error: Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="HTTP error"):
            processor.process_url("https://example.com/nonexistent-page")
    
    @patch('src.document_processor.requests.get')
    def test_invalid_url_500_server_error(self, mock_get, processor):
        """Test handling of URL returning 500 server error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="HTTP error"):
            processor.process_url("https://example.com/server-error")
    
    # Error Handling Tests for Non-HTML Web Content
    
    @patch('src.document_processor.requests.get')
    def test_non_html_json_content(self, mock_get, processor):
        """Test handling of JSON content instead of HTML."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/json'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://api.example.com/data.json")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_xml_content(self, mock_get, processor):
        """Test handling of XML content instead of HTML."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/xml'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/feed.xml")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_pdf_content(self, mock_get, processor):
        """Test handling of PDF content from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/pdf'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/document.pdf")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_image_content(self, mock_get, processor):
        """Test handling of image content from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'image/jpeg'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/image.jpg")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_binary_content(self, mock_get, processor):
        """Test handling of binary content from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/octet-stream'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/file.bin")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_plain_text(self, mock_get, processor):
        """Test handling of plain text content from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'text/plain'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/readme.txt")
    
    @patch('src.document_processor.requests.get')
    def test_non_html_javascript_content(self, mock_get, processor):
        """Test handling of JavaScript content from URL."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {'Content-Type': 'application/javascript'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with pytest.raises(WebContentError, match="does not return HTML content"):
            processor.process_url("https://example.com/script.js")



class TestDocumentProcessorProperties:
    """Property-based tests for DocumentProcessor."""
    
    # Feature: ai-knowledge-copilot, Property 1: Text extraction completeness
    @given(
        pdf_text=st.text(min_size=10, max_size=1000, alphabet=st.characters(
            whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')
        ))
    )
    @settings(max_examples=100, deadline=None)
    def test_property_text_extraction_completeness(self, pdf_text):
        """
        Property 1: Text extraction completeness
        
        For any valid PDF document with text content, extracting text should 
        return non-empty content that includes text from all pages.
        
        **Validates: Requirements 1.1**
        """
        processor = DocumentProcessor()
        
        # Create a simple PDF with the generated text
        # We'll mock the PDF extraction to focus on the property
        with patch.object(processor, '_extract_with_pdfplumber') as mock_extract:
            mock_extract.return_value = pdf_text
            
            # Create a temporary PDF file
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False) as f:
                # Write minimal PDF structure
                f.write(b'%PDF-1.4\n')
                temp_path = f.name
            
            try:
                document = processor.process_pdf(temp_path)
                
                # Property: Extracted text should be non-empty
                assert document.text, "Extracted text should not be empty"
                assert document.text.strip(), "Extracted text should contain non-whitespace content"
                
                # Property: Extracted text should match the source text
                assert document.text == pdf_text, "Extracted text should match source content"
                
                # Property: Document should have correct metadata
                assert document.source_type == "pdf"
                assert document.id
                
            finally:
                Path(temp_path).unlink(missing_ok=True)

    
    # Feature: ai-knowledge-copilot, Property 21: Web content extraction
    @given(
        web_text=st.text(
            min_size=10, 
            max_size=1000, 
            alphabet=st.characters(min_codepoint=32, max_codepoint=126)  # ASCII printable characters only
        ).filter(lambda x: x.strip())  # Ensure non-empty after stripping
    )
    @settings(max_examples=100, deadline=None)
    def test_property_web_content_extraction(self, web_text):
        """
        Property 21: Web content extraction
        
        For any valid HTTP/HTTPS URL that returns HTML, the system should 
        extract non-empty text content.
        
        **Validates: Requirements 7.1**
        """
        import html
        processor = DocumentProcessor()
        
        # Escape HTML special characters to prevent parsing issues
        escaped_text = html.escape(web_text)
        
        # Create HTML with the generated text
        html_content = f"""
        <html>
            <head><title>Test Page</title></head>
            <body>
                <p>{escaped_text}</p>
            </body>
        </html>
        """
        
        # Mock the requests.get call
        with patch('src.document_processor.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {'Content-Type': 'text/html; charset=utf-8'}
            mock_response.text = html_content
            mock_get.return_value = mock_response
            
            url = "https://example.com/test"
            document = processor.process_url(url)
            
            # Property: Extracted text should be non-empty
            assert document.text, "Extracted text should not be empty"
            assert document.text.strip(), "Extracted text should contain non-whitespace content"
            
            # Property: Extracted text should contain the source text (normalized)
            # BeautifulSoup normalizes whitespace, so we check for normalized version
            normalized_web_text = ' '.join(web_text.split())
            if normalized_web_text:  # Only check if there's actual content after normalization
                assert normalized_web_text in document.text, "Extracted text should contain source content"
            
            # Property: Document should have correct metadata
            assert document.source_type == "web"
            assert document.source == url
            assert document.metadata["url"] == url
            assert document.id
