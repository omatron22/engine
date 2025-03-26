-- QmiracTM AI Knowledge Base - Database Schema

-- Document storage
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    document_type TEXT NOT NULL,  -- 'strategy', 'execution', 'market_assessment', etc.
    file_path TEXT,               -- Original file path
    metadata TEXT,                -- JSON metadata about the document
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on document type for faster filtering
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);

-- Embedding storage
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding_vector BLOB NOT NULL,  -- Store binary embedding vectors
    chunk_index INTEGER NOT NULL,
    doc_type TEXT,                   -- Duplicate of document type for faster access
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create indexes for faster embedding retrieval
CREATE INDEX IF NOT EXISTS idx_embeddings_document ON embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk ON embeddings(document_id, chunk_index);

-- Strategic inputs for strategy generation
CREATE TABLE IF NOT EXISTS strategic_inputs (
    id INTEGER PRIMARY KEY,
    risk_tolerance TEXT CHECK(risk_tolerance IN ('High', 'Medium', 'Low')),
    strategic_priorities TEXT,
    strategic_constraints TEXT,
    execution_priorities TEXT,
    execution_constraints TEXT,
    generated_strategy_id INTEGER,  -- Link to generated strategy document if applicable
    user_id TEXT,                   -- Optional user identifier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (generated_strategy_id) REFERENCES documents(id) ON DELETE SET NULL
);

-- User feedback and interaction tracking
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY,
    query TEXT NOT NULL,
    response TEXT NOT NULL, 
    feedback TEXT,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    session_id TEXT,              -- Optional session identifier
    user_id TEXT,                 -- Optional user identifier
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for querying feedback by rating
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON user_feedback(rating);

-- System configuration and settings
CREATE TABLE IF NOT EXISTS system_config (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Usage statistics for monitoring
CREATE TABLE IF NOT EXISTS usage_stats (
    id INTEGER PRIMARY KEY,
    action_type TEXT NOT NULL,   -- 'query', 'strategy_generation', 'document_upload', etc.
    details TEXT,                -- JSON details about the action
    duration_ms INTEGER,         -- Time taken in milliseconds
    successful BOOLEAN,          -- Whether the action was successful
    error_message TEXT,          -- Error message if unsuccessful
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to update documents.updated_at when a row is updated
CREATE TRIGGER IF NOT EXISTS update_documents_timestamp
AFTER UPDATE ON documents
BEGIN
    UPDATE documents SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Trigger to update system_config.updated_at when a row is updated
CREATE TRIGGER IF NOT EXISTS update_system_config_timestamp
AFTER UPDATE ON system_config
BEGIN
    UPDATE system_config SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Insert default system configuration values
INSERT OR IGNORE INTO system_config (key, value, description) VALUES
    ('default_model', 'deepseek-coder:reasoning', 'Default LLM model to use'),
    ('embedding_model', 'all-MiniLM-L6-v2', 'Model used for generating embeddings'),
    ('chunk_size', '500', 'Default chunk size for text splitting'),
    ('chunk_overlap', '50', 'Default overlap between chunks'),
    ('top_k_results', '5', 'Default number of results to retrieve'),
    ('version', '0.1.0', 'Knowledge base schema version');