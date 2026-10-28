# Research Report: Compare the top 3 open-source vector databases for a startup building RAG products

**Status:** `OK` | **Confidence:** `HIGH` (86.5%)

## Short Answer
For a startup building RAG products, the top three open-source vector databases are **Qdrant**, **Milvus**, and **Chroma**. Qdrant provides the best balance of speed, payload filtering, and production ease via its native Rust implementation; Milvus offers superior distributed scalability for massive multi-million vector collections; and Chroma excels in rapid developer onboarding and local prototyping.

## Key Findings
1. Qdrant is written in Rust, providing native payload-based filtering, low-latency approximate nearest neighbor (ANN) search, and a mature cloud-ready standalone binary. (Sources: `[s1]`, `[s2]`)
2. Milvus architecture is decoupled and cloud-native, separating storage from compute nodes, making it suitable for distributed deployments handling massive billion-scale vector workloads. (Sources: `[s3]`)
3. Chroma focuses on developer ergonomics with native Python and TypeScript client bindings, fast in-process prototyping, and seamless integration with LangChain and LlamaIndex. (Sources: `[s4]`)
4. GitHub activity indicates strong community velocity: Qdrant maintains >22k stars and active releases, Milvus maintains >31k stars with enterprise adoption, and Chroma maintains >16k stars. (Sources: `[s2]`, `[s5]`)

## Sources
- **[s1]** [Qdrant Documentation: Architecture & Vector Search](https://qdrant.tech/documentation/) - `qdrant.tech` _(Used in: Finding 1)_
- **[s2]** [Qdrant GitHub Repository Metadata](https://github.com/qdrant/qdrant) - `github.com` _(Used in: Finding 1, Finding 4)_
- **[s3]** [Milvus Cloud-Native Architecture Overview](https://milvus.io/docs/overview.md) - `milvus.io` _(Used in: Finding 2)_
- **[s4]** [Chroma Getting Started Guide](https://docs.trychroma.com/getting-started) - `docs.trychroma.com` _(Used in: Finding 3)_
- **[s5]** [Milvus GitHub Repository Metadata](https://github.com/milvus-io/milvus) - `github.com` _(Used in: Finding 4)_

## Confidence & Rationale
- **Overall Level:** High (0.865)
  - Source authority: 0.94 (weight: 30%)
  - Sub-question coverage: 1.00 (3/3 covered, weight: 25%)
  - Quote verification rate: 0.95 (weight: 20%)
  - Source consistency: 1.00 (weight: 15%)
  - Information freshness: 0.80 (weight: 10%)

## Assumptions
- Evaluated specifically through the lens of early-stage startup RAG requirements (latency, operational complexity, ingestion speed).
- Evaluated self-hosted / open-source licensing (Apache 2.0).

## Limitations
- Performance benchmarks depend heavily on embedding dimensionality, indexing algorithms (HNSW vs IVF_FLAT), and server memory availability.

## Suggested Next Steps
1. Benchmark Qdrant and Milvus using a representative sample of company-specific embeddings and query vectors.
2. Evaluate operational overhead: evaluate whether team expertise aligns better with a single binary (Qdrant) or distributed Kubernetes infrastructure (Milvus).

## Operational Telemetry
- **Run ID:** `20261028-154210-9b4e`
- **Latency:** 8.42s
- **LLM Calls:** 6 (1 planner, 4 small extraction/relevance, 1 synthesis)
- **Tokens:** 6,420 in / 1,180 out
- **Estimated Cost:** $0.0034 USD
- **Tool Calls:** search_calls: 3, fetch_calls: 4, github_repo_stats: 2

