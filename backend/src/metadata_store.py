"""
Metadata storage using SQLite for document and chunk metadata.

This module provides persistent storage for document and chunk metadata,
enabling efficient retrieval and management of indexed content.
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class MetadataStore:
    """
    SQLite-based metadata storage for documents and chunks.
    
    This class manages document and chunk metadata with support for:
    - Document tracking with unique IDs
    - Chunk metadata with foreign key relationships
    - Cascade deletion of chunks when documents are deleted
    - Efficient querying and listing operations
    
    Validates Requirements 2.3, 11.1, 11.2, 11.5
    """
    
    def __init__(self, db_path: str = "./data/metadata.db"):
        """
        Initialize the metadata store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Ensure the directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database connection and schema
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema with documents and chunks tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create documents table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                source_type TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Create chunks table with foreign key to documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            )
        """)
        
        # Create index on document_id for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id 
            ON chunks(document_id)
        """)
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add a document to the metadata store.
        
        Args:
            doc_id: Unique document identifier
            metadata: Document metadata including source, source_type, created_at, etc.
        
        Validates Requirement 2.3 (metadata storage)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract required fields
        source = metadata.get('source', '')
        source_type = metadata.get('source_type', 'pdf')
        created_at = metadata.get('created_at', datetime.now().isoformat())
        
        # Convert datetime to string if needed
        if isinstance(created_at, datetime):
            created_at = created_at.isoformat()
        
        # Store remaining metadata as JSON
        remaining_metadata = {k: v for k, v in metadata.items() 
                            if k not in ['source', 'source_type', 'created_at']}
        metadata_json = json.dumps(remaining_metadata)
        
        cursor.execute("""
            INSERT OR REPLACE INTO documents (id, source, source_type, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (doc_id, source, source_type, metadata_json, created_at))
        
        conn.commit()
        conn.close()

    def add_chunk(self, chunk_id: str, metadata: Dict[str, Any]) -> None:
        """
        Add a chunk to the metadata store.
        
        Args:
            chunk_id: Unique chunk identifier
            metadata: Chunk metadata including document_id, text, chunk_index, etc.
        
        Validates Requirement 2.3 (chunk metadata completeness)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Extract required fields
        document_id = metadata.get('document_id', '')
        text = metadata.get('text', '')
        
        # Store remaining metadata as JSON
        remaining_metadata = {k: v for k, v in metadata.items() 
                            if k not in ['document_id', 'text']}
        metadata_json = json.dumps(remaining_metadata)
        
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (id, document_id, text, metadata)
            VALUES (?, ?, ?, ?)
        """, (chunk_id, document_id, text, metadata_json))
        
        conn.commit()
        conn.close()
    
    def get_chunk_metadata(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific chunk.
        
        Args:
            chunk_id: Unique chunk identifier
        
        Returns:
            Dictionary containing chunk metadata, or None if not found
        
        Validates Requirement 2.3 (metadata retrieval)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, document_id, text, metadata
            FROM chunks
            WHERE id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        # Reconstruct metadata dictionary
        chunk_id, document_id, text, metadata_json = row
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Add core fields to metadata
        metadata['id'] = chunk_id
        metadata['document_id'] = document_id
        metadata['text'] = text
        
        return metadata
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """
        Retrieve all chunk IDs for a specific document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            List of chunk IDs belonging to the document
        
        Validates Requirement 11.5 (metadata-vector consistency)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id FROM chunks
            WHERE document_id = ?
            ORDER BY id
        """, (doc_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [row[0] for row in rows]
    
    def delete_document(self, doc_id: str) -> None:
        """
        Delete a document and all its associated chunks.
        
        Uses CASCADE deletion to automatically remove all chunks
        when a document is deleted.
        
        Args:
            doc_id: Document identifier to delete
        
        Validates Requirement 11.1 (document deletion completeness)
        """
        conn = sqlite3.connect(self.db_path)
        
        # Enable foreign key constraints (required for CASCADE)
        conn.execute("PRAGMA foreign_keys = ON")
        
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM documents
            WHERE id = ?
        """, (doc_id,))
        
        conn.commit()
        conn.close()
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all indexed documents with their metadata.
        
        Returns:
            List of dictionaries containing document metadata
        
        Validates Requirement 11.2 (document listing completeness)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, source, source_type, metadata, created_at
            FROM documents
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            doc_id, source, source_type, metadata_json, created_at = row
            
            # Parse metadata JSON
            metadata = json.loads(metadata_json) if metadata_json else {}
            
            # Build document dictionary
            doc = {
                'id': doc_id,
                'source': source,
                'source_type': source_type,
                'created_at': created_at,
                **metadata
            }
            documents.append(doc)
        
        return documents
    
    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve metadata for a specific document.
        
        Args:
            doc_id: Document identifier
        
        Returns:
            Dictionary containing document metadata, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, source, source_type, metadata, created_at
            FROM documents
            WHERE id = ?
        """, (doc_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row is None:
            return None
        
        doc_id, source, source_type, metadata_json, created_at = row
        
        # Parse metadata JSON
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Build document dictionary
        doc = {
            'id': doc_id,
            'source': source,
            'source_type': source_type,
            'created_at': created_at,
            **metadata
        }
        
        return doc
    
    def clear_all(self) -> None:
        """
        Clear all documents and chunks from the metadata store.
        
        This is useful for testing and resetting the system.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM documents")
        cursor.execute("DELETE FROM chunks")
        
        conn.commit()
        conn.close()
    
    def close(self) -> None:
        """
        Close the database connection.
        
        Note: This implementation uses connection-per-operation,
        so this method is provided for API compatibility but doesn't
        maintain a persistent connection.
        """
        pass
