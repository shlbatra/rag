import pydantic

# Pydantic models for custom types used in RAG functions and validation

# Chunk pdf doc
class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

# Insert result to Vector DB
class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

# Search result from Vector DB
class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]

# Result to User
class RAQQueryResult(pydantic.BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int