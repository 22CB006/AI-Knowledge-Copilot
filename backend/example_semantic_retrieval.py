"""
Example script demonstrating semantic retrieval functionality.

This script shows how to use the SemanticRetriever to perform
semantic search over indexed documents.
"""
import numpy as np
from src.semantic_retriever import SemanticRetriever
from src.vector_store import VectorStore
from src.embedder import EmbeddingGenerator
from src.metadata_store import MetadataStore


def main():
    """Demonstrate semantic retrieval with example data."""
    print("=== Semantic Retrieval Example ===\n")
    
    # Initialize components
    print("1. Initializing components...")
    embedder = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    dimension = embedder.get_embedding_dimension()
    vector_store = VectorStore(dimension=dimension)
    metadata_store = MetadataStore(db_path="./data/example_metadata.db")
    
    # Create retriever
    retriever = SemanticRetriever(vector_store, embedder, metadata_store)
    print(f"   ✓ Initialized with embedding dimension: {dimension}\n")
    
    # Add example document
    print("2. Adding example document...")
    doc_id = "ml_basics"
    metadata_store.add_document(doc_id, {
        'source': 'machine_learning_guide.pdf',
        'source_type': 'pdf'
    })
    
    # Add example chunks
    example_chunks = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data.",
        "Supervised learning uses labeled data to train models for prediction tasks.",
        "Unsupervised learning finds patterns in unlabeled data through clustering and dimensionality reduction.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
        "Natural language processing enables computers to understand and generate human language.",
        "Computer vision allows machines to interpret and analyze visual information from images and videos.",
        "Reinforcement learning trains agents to make decisions through trial and error with rewards.",
        "Feature engineering is the process of selecting and transforming variables for machine learning models."
    ]
    
    chunk_ids = []
    embeddings_list = []
    
    for i, text in enumerate(example_chunks):
        chunk_id = f"chunk_{i}"
        chunk_ids.append(chunk_id)
        
        # Add chunk metadata
        metadata_store.add_chunk(chunk_id, {
            'document_id': doc_id,
            'text': text,
            'chunk_index': i,
            'page_number': i + 1
        })
        
        # Generate embedding
        embedding = embedder.embed_text(text)
        embeddings_list.append(embedding)
    
    # Add embeddings to vector store
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    vector_store.add_vectors(embeddings_array, chunk_ids)
    print(f"   ✓ Added {len(example_chunks)} chunks to the index\n")
    
    # Perform semantic retrieval
    print("3. Performing semantic retrieval...\n")
    
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "Tell me about reinforcement learning",
        "What is NLP?"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        print("-" * 60)
        
        # Retrieve top 3 results
        results = retriever.retrieve(query, k=3)
        
        if not results:
            print("   No results found.\n")
            continue
        
        for i, result in enumerate(results, 1):
            print(f"\n   Result {i}:")
            print(f"   Score: {result.score:.4f}")
            print(f"   Document: {result.document_name}")
            print(f"   Page: {result.page_number}")
            print(f"   Text: {result.chunk.text[:100]}...")
        
        print("\n")
    
    # Demonstrate retrieve_with_scores
    print("4. Using retrieve_with_scores method...\n")
    query = "neural networks and AI"
    print(f"Query: '{query}'")
    print("-" * 60)
    
    results_with_scores = retriever.retrieve_with_scores(query, k=3)
    
    for i, (chunk, score) in enumerate(results_with_scores, 1):
        print(f"\n   Result {i} (Score: {score:.4f}):")
        print(f"   {chunk.chunk.text[:80]}...")
    
    print("\n\n=== Example Complete ===")
    
    # Cleanup
    metadata_store.clear_all()


if __name__ == "__main__":
    main()
