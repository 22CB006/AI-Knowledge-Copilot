"""
Example script demonstrating the reranking functionality.

This script shows how to use the Reranker class to improve retrieval results
by reordering chunks based on relevance to a specific query.
"""
from src.reranker import Reranker
from src.models import RetrievedChunk, Chunk


def main():
    """Demonstrate reranking functionality."""
    print("=" * 80)
    print("Reranking Engine Demo")
    print("=" * 80)
    
    # Initialize the reranker
    print("\n1. Initializing Reranker with cross-encoder model...")
    reranker = Reranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"   Model loaded: {reranker.model_name}")
    
    # Create sample chunks (simulating retrieval results)
    print("\n2. Creating sample retrieved chunks...")
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "The weather forecast predicts sunny skies and temperatures around 75 degrees.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Pizza is a popular Italian dish consisting of dough, sauce, cheese, and toppings.",
        "Natural language processing allows computers to understand and generate human language.",
        "Reinforcement learning trains agents through trial and error using rewards.",
        "The stock market experienced volatility today with mixed trading results."
    ]
    
    chunks = []
    for i, text in enumerate(sample_texts):
        chunk = Chunk(
            id=f"chunk_{i}",
            document_id=f"doc_{i % 3}",
            text=text,
            embedding=None,
            metadata={
                'chunk_index': i,
                'page_number': i + 1,
                'document_id': f"doc_{i % 3}"
            }
        )
        
        # Simulate initial retrieval scores (not necessarily in relevance order)
        retrieved_chunk = RetrievedChunk(
            chunk=chunk,
            score=0.5 + (i * 0.05),  # Arbitrary initial scores
            document_name=f"document_{i % 3}.pdf",
            page_number=i + 1
        )
        
        chunks.append(retrieved_chunk)
    
    print(f"   Created {len(chunks)} sample chunks")
    
    # Define a query
    query = "What is machine learning and how does it relate to AI?"
    print(f"\n3. Query: '{query}'")
    
    # Show initial retrieval order
    print("\n4. Initial retrieval order (by initial scores):")
    for i, chunk in enumerate(chunks, 1):
        print(f"   {i}. [Score: {chunk.score:.3f}] {chunk.chunk.text[:70]}...")
    
    # Rerank the chunks
    print("\n5. Reranking chunks using cross-encoder model...")
    top_n = 5
    reranked_chunks = reranker.rerank(query, chunks, top_n=top_n)
    
    # Show reranked results
    print(f"\n6. Top {top_n} chunks after reranking:")
    for i, chunk in enumerate(reranked_chunks, 1):
        print(f"   {i}. [Score: {chunk.score:.3f}] {chunk.chunk.text[:70]}...")
    
    # Demonstrate score_pairs method
    print("\n7. Demonstrating score_pairs method:")
    test_texts = [
        "Machine learning is a type of AI.",
        "The weather is nice today.",
        "Deep learning is a subset of machine learning."
    ]
    
    scores = reranker.score_pairs(query, test_texts)
    print(f"   Query: '{query}'")
    for text, score in zip(test_texts, scores):
        print(f"   Score: {score:8.3f} | Text: {text}")
    
    # Verify metadata preservation
    print("\n8. Verifying metadata preservation:")
    for chunk in reranked_chunks[:3]:
        print(f"   Chunk ID: {chunk.chunk.id}")
        print(f"   Document: {chunk.document_name}")
        print(f"   Page: {chunk.page_number}")
        print(f"   Metadata: {chunk.chunk.metadata}")
        print()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    
    # Show the improvement
    print("\n9. Analysis:")
    print("   Notice how reranking moved the most relevant chunks (about ML/AI)")
    print("   to the top, while less relevant chunks (weather, pizza, stocks)")
    print("   were ranked lower or excluded from the top-5 results.")
    print("\n   This demonstrates how reranking improves retrieval quality by")
    print("   using a more sophisticated cross-encoder model that jointly")
    print("   processes the query and document text.")


if __name__ == "__main__":
    main()
