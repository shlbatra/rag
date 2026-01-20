

Run fastapi app
uv run uvicorn main:app

Inngest
Orchestration tool, logging, observability
Covers triggers (event or webhook or schedule), flow control(function distributed in time with concurrency and throttling) and steps(retriable checkpoint)
command -> 
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery (Run dev server connect app on prt 8000 at api/inngest endpoint)
local dev server
Server running between fastapi and client 
ex. upload pdf - request send to ingest serve - ingest function - call function (takes care of calling, logging, rate limiting, etc)

VectorDB (Qdrant)
Store data in vector format, text convert to embeddings - search vector based on similarity
Run qdrant locally ->
docker run -d --name qdrantRagDb -p 6333:6333 -v "./qdrant_storage:/qdrant/storage" qdrant/qdrant

LlamaIndex
load pdf, parse and pass to Qdrant

OpenAI
LLMmodels

Streamlit 
Frontend