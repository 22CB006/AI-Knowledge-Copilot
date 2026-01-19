"""
Example script demonstrating the DocumentIndexer usage.

This script shows how to:
1. Initialize all required components
2. Index a document (PDF or web content)
3. Retrieve document information
4. List all indexed documents
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.document_processor import DocumentProcessor
from src.chunker import TextChunker
from src.embedder import EmbeddingGenerator
from src.vector_store import VectorStore
from src.metadata_store import MetadataStore
from src.document_indexer import DocumentIndexer


def main():
    """Demonstrate document indexing pipeline."""
    
    print("=" * 60)
    print("Document Indexing Pipeline Example")
    print("=" * 60)
    
    # Step 1: Initialize all components
    print("\n1. Initializing components...")
    
    document_processor = DocumentProcessor()
    chunker = TextChunker(chunk_size=512, overlap=50)
    embedder = EmbeddingGenerator("all-MiniLM-L6-v2")
    
    # Get embedding dimension for vector store
    dimension = embedder.get_embedding_dimension()
    print(f"   - Embedding dimension: {dimension}")
    
    vector_store = VectorStore(dimension=dimension)
    metadata_store = MetadataStore("./data/example_metadata.db")
    
    # Create the indexer
    indexer = DocumentIndexer(
        document_processor=document_processor,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        metadata_store=metadata_store
    )
    
    print("   ✓ All components initialized")
    
    # Step 2: Index a web page (example)
    print("\n2. Indexing a web page...")
    
    try:
        # Example: Index Python documentation
        url = "https://docs.python.org/3/tutorial/introduction.html"
        print(f"   - URL: {url}")
        
        doc_id = indexer.index_url(url)
        print(f"   ✓ Document indexed with ID: {doc_id}")
        
        # Step 3: Get document information
        print("\n3. Retrieving document information...")
        doc_info = indexer.get_document_info(doc_id)
        
        if doc_info:
            print(f"   - Document ID: {doc_info['id']}")
            print(f"   - Source: {doc_info['source']}")
            print(f"   - Source Type: {doc_info['source_type']}")
            print(f"   - Chunk Count: {doc_info['chunk_count']}")
            print(f"   - Created At: {doc_info['created_at']}")
        
        # Step 4: List all documents
        print("\n4. Listing all indexed documents...")
        documents = indexer.list_documents()
        print(f"   - Total documents: {len(documents)}")
        
        for i, doc in enumerate(documents, 1):
            print(f"\n   Document {i}:")
            print(f"     - ID: {doc['id']}")
            print(f"     - Source: {doc['source'][:50]}...")
            print(f"     - Chunks: {doc['chunk_count']}")
        
        # Step 5: Verify vector storage
        print("\n5. Verifying vector storage...")
        vector_count = vector_store.get_vector_count()
        print(f"   - Total vectors in store: {vector_count}")
        
        print("\n" + "=" * 60)
        print("✓ Example completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        print("\nNote: This example requires internet connection to fetch web content.")
        print("You can also test with a local PDF file using:")
        print("  doc_id = indexer.index_pdf('/path/to/your/file.pdf')")


if __name__ == "__main__":
    main()
