---
title: "Why Naive RAG Fails in Healthcare and What Matters in Clinical RAG"
date: 2026-04-28T14:30:00+05:30
description: "An in-depth guide on why naive Retrieval-Augmented Generation (RAG) fails in clinical healthcare systems, and the production-grade design patterns (parent-child chunking, cross-encoder reranking, and dynamic LLMOps observability) needed to fix them."
summary: "An in-depth guide on why naive Retrieval-Augmented Generation (RAG) fails in clinical healthcare systems, and the production-grade design patterns (parent-child chunking, cross-encoder reranking, and dynamic LLMOps observability) needed to fix them."
keywords: ["Clinical RAG", "Healthcare AI", "Naive RAG Failures", "Parent-Child Chunking", "Cross-Encoder Reranking", "LLMOps", "Vector Collapse"]
tags: ["Clinical RAG", "Healthcare", "NLP", "Vector Databases", "Inference", "LLMOps"]
draft: false
---

In a sandbox, you load a handful of clean PDFs, slice them at 512 tokens, write vectors to a local FAISS index, and wire the components together. The outputs look reasonable. In a production clinical deployment, that same architecture breaks before a single query reaches the model. Nothing in the logs tells you where.

> **A Typical Clinical Failure Cascade:** A patient with chronic kidney disease presents with acute pain. The hospital's naive RAG system processes a scanned medical history fax. The OCR engine silently drops a low-contrast page listing a Stage 4 renal impairment diagnosis. Without this parent context, the retriever fails to surface the impairment chunk, and the LLM confidently generates a recommendation for a high-dose NSAID like Ibuprofen. The downstream result is an avoidable, severe acute-on-chronic kidney injury. The model did not fail, but the pipeline failed long before inference.

This post documents four concrete failure modes that appear in clinical RAG systems, analyzing each through a disciplined engineering lens: Failure, Root Cause, Architectural Fix, Trade-offs, and Production Implications.

---

## The Data Problem Comes Before the Model Problem

Every failure mode discussed below originates from the same root cause: the data environment in healthcare is hostile by default.

Clinical information is split across hospitals, outpatient clinics, radiology systems, and insurance platforms. None of these systems share a common format. A patient's primary diagnosis may sit in a typed PDF at one facility, their comorbidity list in a scanned fax from another, and their lab values in an Excel export from a third. This is not an edge case. It is the standard operational reality.

The direct engineering implication: no amount of model quality compensates for incomplete indexing. A state-of-the-art LLM reasoning over a contaminated or partial vector index will produce confident, plausible, and clinically wrong answers. The model has no visibility into what was silently dropped at ingestion. It can only reason over what the retriever surfaces.

Every architectural decision in a clinical RAG system must be evaluated against this constraint first.

---

## Failure Mode 1: Ingestion Collapse on Non-Digital Records

### 1. The Failure
Standard PDF parsing libraries (such as `pdfplumber`, `PyPDF2`, or LangChain's default `PyPDFLoader`) extract text from a PDF's content stream. When that stream is absent, they return an empty string. No exception is raised. The page counter increments. The embedding call receives an empty string and encodes it into a near-zero vector that is written into the index.

Clinical records routinely contain pages with no selectable text layer: scanned referral letters, photographed pathology reports, faxed discharge summaries, and legacy records digitized at low resolution. At scale, every one of these pages becomes a silent void in the index. A clinician querying for a patient's prior surgical history may receive a "no relevant information found" response precisely because that history lived on a scanned page the ingestion pipeline discarded without logging anything.

### 2. The Root Cause
Standard parsers are built to read digital character codes directly. They lack any integrated computer vision layer. Because a scanned image contains no character streams, the parser has nothing to read. The failure is completely silent because returning an empty string is a valid parser return state rather than a system exception.

### 3. The Architectural Fix: Dual-Layer Adaptive OCR
Production ingestion pipelines for clinical documents require a two-stage extraction strategy:

* **Stage 1 (PyMuPDF / fitz):** Process every page with PyMuPDF first. It is fast, handles digital PDFs with high fidelity, preserves reading order, and correctly parses multi-column clinical layouts.
* **Stage 2 (Adaptive EasyOCR Fallback):** When Stage 1 returns a blank result, the pipeline rasterizes the page to an image and passes it through EasyOCR. If the extraction quality remains low, an adaptive DPI loop is triggered: render at 2x DPI first, and if the result is still insufficient, re-render at 3x DPI.

```python
# Dual-layer ingestion pattern
page_text = extract_with_pymupdf(page)

if not page_text.strip():
    image = rasterize_page(page, dpi=200)
    page_text = run_easyocr(image)

    if not page_text.strip():
        image = rasterize_page(page, dpi=300)  # Adaptive retry
        page_text = run_easyocr(image)

# Flag the page, never silently discard it
```

Every chunk produced from an OCR-processed page carries an `ocr_used` boolean in its metadata. This single field makes ingestion quality observable downstream.

### 4. The Trade-offs
Adding local OCR introduces massive CPU and GPU resource requirements for production throughput. While PyMuPDF processes pages in milliseconds, OCR engines require several seconds per page. This introduces a scaling bottleneck at the ingestion layer, requiring asynchronous job workers (such as Celery) to avoid blocking the primary application thread.

### 5. The Production Implication: Cloud APIs vs. Self-Hosted VPC
Managed cloud services like AWS Textract and Google Cloud Document AI offer highly accurate document OCR and handle complex clinical tables natively. For many deployments, they represent the lowest operational overhead.

However, data sovereignty is the primary constraint. Sending raw patient medical faxes to a third-party managed API traverses a security boundary. For systems bound by strict regulatory constraints or national data residency requirements, this is often a non-starter. 

In these zero-tolerance environments, hosting a local PyMuPDF + EasyOCR extraction stack inside a secure, private Virtual Private Cloud (VPC) is mandatory. The operational burden is high, but it guarantees that Protected Health Information (PHI) never leaves your private network boundary.

---

## Failure Mode 2: Fixed-Size Chunking and Semantic Dilution

### 1. The Failure
Dense embedding models compress the semantic content of a text span into a single vector through mean pooling over token representations. When a fixed-size chunker slices a continuous clinical narrative into rigid blocks (e.g., 512 tokens), it routinely cuts across highly specific medical descriptions.

Consider this clinical statement: *"Patient presents with Type 2 Diabetes Mellitus (ICD-10: E11.9), hypertensive crisis (ICD-10: I16.0), and early-stage chronic kidney disease (ICD-10: N18.3)."*

A fixed-size chunker will frequently split this sentence across a boundary, isolating the diagnosis code from its qualifying condition. More critically, rare but vital entities (like specific ICD-10 medical billing codes or rare drug names) get semantically averaged out during vector generation. The embedding vector gets dominated by high-frequency surrounding terms, leading to semantic dilution. When a clinician queries for a specific condition, the retriever fails to surface the relevant context.

### 2. The Root Cause
Embedding models compress large blocks of text into a fixed-dimensional space (e.g., 384 or 768 float dimensions). Mean pooling calculates the average value of all token embeddings in the chunk. Because specific clinical entities (like exact drug names or billing codes) are mathematically rare compared to common clinical grammar ("patient presents with", "history of"), their high-signal vector representations are effectively diluted to near-zero significance.

### 3. The Architectural Fix: Parent-Child Chunking & Multi-Pathway Retrieval
To resolve this, the clinical pipeline must separate the retrieval unit from the generation unit:

* **Child Chunks (300 to 500 characters):** Calibrated to isolate single clinical statements (e.g., a single diagnosis paired with its ICD code). Only these small, high-signal child chunks are indexed in the vector database.
* **Parent Chunks (1,500 to 2,000 characters):** Stored separately in an in-process or distributed key-value store (such as Redis or MongoDB). Each child chunk carries a `parent_id` reference in its metadata pointing back to its parent.
* **Retrieval Expansion:** When a child chunk is matched, the orchestrator dynamically resolves its `parent_id` and substitutes the larger, enclosing parent context before passing it to the LLM.

```python
# Parent-Child retrieval expansion
child_results = vector_db.similarity_search(query, k=10)

contexts_for_llm = []
for child in child_results:
    parent_id = child.metadata["parent_id"]
    parent_chunk = document_store[parent_id]  # Retrieve expanded parent context
    contexts_for_llm.append(parent_chunk)
```

To counter semantic dilution further, two additional retrieval pathways should run in parallel:
* **Sparse BM25 Indexing:** A keyword-based index running alongside the vector store. The final retrieval score is calculated using Reciprocal Rank Fusion (RRF) to combine dense semantic relevance and sparse exact-token matches. This ensures rare clinical codes are never dropped.
* **Relational EHR Lookups:** Structured, longitudinal numerical data (like a patient's HbA1c history over the last 12 months) is fetched directly from relational Electronic Health Record (EHR) databases via SQL queries, rather than using vector approximations.

### 4. The Trade-offs
Storing separate child vectors and parent text blocks doubles the storage footprint and increases indexing complexity. Resolving parent IDs adds an extra retrieval hop, introducing approximately 10 to 20 milliseconds of latency to the query path.

### 5. The Production Implication: Alternatives and Temporal Calibration
While Parent-Child chunking is a standard production design pattern, engineers must evaluate it against other retrieval architectures:
* **Late Chunking & ColBERT:** ColBERT uses late interaction over token-level embeddings, preserving fine-grained entity signals without rigid boundaries, though it demands higher index storage overhead.
* **RAPTOR:** Generates hierarchical summaries of clinical records, which is excellent for high-level summaries but overly complex for specific entity lookup.

Furthermore, clinical history is fundamentally longitudinal. A medical chart from yesterday is vastly more relevant than one from five years ago. To handle this, the ranking algorithm must implement **Temporal decay weighting**:

**Temporal Score = RRF Score * e^(-λt)**

Where:
* **t** represents the age of the document in days relative to the query.
* **λ (lambda)** is a decay coefficient that must be calibrated by domain: high decay (volatile inpatient vital logs) vs. near-zero decay (stable baseline surgical records or permanent allergy profiles).

---

## Failure Mode 3: The Lost-in-the-Middle Attention Bias

### 1. The Failure
When the retriever surfaces the top-k chunks, naive pipelines concatenate them in raw vector similarity order before passing them to the LLM. Under high query volume, highly relevant clinical details (such as a specific medication allergy) frequently end up buried in the middle of a large assembled context block. The LLM confidently generates a response that completely ignores this critical middle evidence, leading to clinical errors.

### 2. The Root Cause
Transformer models do not attend uniformly across a large context window. Attention maps consistently demonstrate primacy and recency bias: models give disproportionate weight to content placed at the absolute start and end of the prompt, with a sharp drop in attention for content in the middle. Furthermore, cosine similarity in bi-encoder retrieval measures geometric proximity, not clinical relevance, meaning less relevant general text often outranks specific clinical exceptions.

### 3. The Architectural Fix: Cross-Encoder Reranking
The RAG orchestrator must introduce a joint-attention **Cross-Encoder Reranker** (such as FlashRank or Cohere Rerank) immediately after bi-encoder retrieval:

```
Vector Store (bi-encoder, k=10) ➔ Cross-Encoder Rerank ➔ top-3 ➔ Parent Expansion ➔ LLM
```

Cross-encoders evaluate the query and each candidate passage jointly, calculating an actual relevance score rather than comparing isolated vectors. The top reranked parent chunks are assembled with the highest-relevance passage placed first, directly aligning clinical signal with the position of maximum LLM attention.

Additionally, a strict context token budget (typically capped between 4,000 to 6,000 tokens) must be enforced using tokenizers like `tiktoken` to prune low-signal text and prevent prompt bloat.

### 4. The Trade-offs
Joint query-passage attention is computationally expensive. Running a cross-encoder model adds substantial query-path latency, typically between 150 to 250 milliseconds depending on host hardware.

### 5. The Production Implication: PHI Redaction & Security Boundaries
If your architecture relies on cloud-based managed reranking APIs (such as Cohere or Jina), you face a serious security boundary constraint. Traversing external networks with raw clinical chunks containing patient identifiers violates data privacy mandates.

To mitigate this, a production-grade pipeline must implement a local **PHI Redaction Layer** (using toolkits like Microsoft Presidio or custom regex tokenizers) to dynamically mask names, dates, and locations before external egress. If using local cross-encoder models within a private VPC, this step is bypassed, but strict Role-Based Access Control (RBAC) and detailed access logging are still required.

---

## Failure Mode 4: Subjective Evaluation in a Zero-Tolerance Domain

### 1. The Failure
Prototypes rely on manual "eyeball checks" to evaluate model responses. In a clinical production setting, manual evaluation is non-deterministic, fails to scale, and cannot detect silent retrieval drift after document ingestion events. More critically, human review cannot distinguish between a **retrieval failure** (correct reasoning over wrong chunks) and a **generation failure** (hallucination over correct chunks), leaving engineers blind when debugging.

### 2. The Root Cause
RAG output quality is not a single dimension; it is a multi-variable state. Human reviews are highly subjective and introduce inconsistent quality baselines. Without isolating the distinct layers of RAG performance, manual evaluations cannot produce actionable metrics.

### 3. The Architectural Fix: Deterministic LLM-as-Judge
Every query response must be evaluated in real-time by an automated, deterministic judge model (such as `llama-3-8b` run at `temperature=0`). The judge evaluates the query, the retrieved context, and the generated response to output three specific metrics on a [0, 1] scale:

* **Faithfulness:** Does the response rely *only* on the retrieved context? A low score indicates hallucination.
* **Answer Relevance:** Does the response directly address the query?
* **Context Precision:** Did the retrieved context contain the necessary information?

```python
# Composite grading logic
# GRADE_THRESHOLDS = {"A": 0.85, "B": 0.70, "C": 0.50}
composite = (faithfulness + answer_relevance + context_precision) / 3
grade = "A" if composite >= 0.85 else "B" if composite >= 0.70 else "C" if composite >= 0.50 else "F"
```

Low-scoring queries are automatically flagged with comprehensive telemetry (chunk IDs, reranker scores, OCR metadata) to enable immediate root-cause tracking.

### 4. The Trade-offs
Evaluating every query synchronously adds 300 to 400 milliseconds of latency to the user path. At scale, this creates a severe bottleneck. The trade-off is resolved by decoupling evaluation: return the primary response immediately, publish the evaluation job to an asynchronous task queue (such as Celery with Redis), and process the metrics in the background.

### 5. The Production Implication: Bias Mitigation & Clinical Fallbacks
Automated judges are highly capable, but they are not infallible. Research indicates they are susceptible to **self-preference bias**, **positional bias**, and **verbosity drift**. Therefore, an LLM-as-Judge framework must augment, not replace, periodic human clinical review. It acts as an automated triage system to flag outliers for human eyes.

Additionally, under low evaluation scores, the system must trigger a **Clinical Fallback pathway** (safe abstention):

If the composite score falls below T_clinical < 0.70 or if the Faithfulness score drops below 0.80:
1. **Redact:** Intercept the generated answer to prevent it from reaching the UI.
2. **Abstain:** Display a pre-formatted, safe fallback message: *"I am unable to confidently summarize this clinical context based on the retrieved records. Please review the patient's primary documents directly."*
3. **Escalate:** Publish the failed transaction to a clinical review queue for manual administrative inspection.

### 6. The Compliance Blueprint: Citation and Clinician-in-the-Loop
For high-stakes clinical deployments, clinical accuracy is only half the battle. Regulatory bodies (such as the FDA under Software as a Medical Device, or SaMD, frameworks) and clinical safety standards demand extreme auditability:

* **Provenance and Citation Tracking:** A production clinical RAG pipeline must never return a plain text answer. Every generated diagnostic assertion or therapeutic recommendation must be decorated with explicit inline citations. These citations map directly to the source document's filename, page number, and original OCR text snippet in the metadata database, giving physicians the power of instant auditability.
* **Clinician-in-the-Loop Oversight:** While automated LLM judges flag quality regressions, high-risk clinical decisions require human-in-the-loop oversight. Queries flagging a low-confidence score are routed to clinical triage queues where physicians can manually review, override, and log modifications to the RAG outputs, satisfying rigorous HIPAA documentation guidelines.

---

## Standard Production Execution Trace

Establishing a clear execution lifecycle is critical for keeping track of performance. Below is a comprehensive reference blueprint for a production-grade execution trace inside a `get_response()` loop:

```
[Incoming Query] ➔ [Phase 0: SHA-256 Cache Check] ➔ (Cache Hit ➔ Return Response)
                                  ↓ (Cache Miss)
                       [Phase 1: Dense Vector Retrieval (k=10)]
                                  ↓
                       [Phase 2: Relevance Reranking (top-3)]
                                  ↓
                       [Phase 2.5: Context Expansion (Parent Chunks)]
                                  ↓
                       [Phase 2.6: Token Budgeting (max 6000)]
                                  ↓
                       [Phase 3: Token Auditing]
                                  ↓
                       [Phase 3.5: Intelligent Model Routing]
                                  ↓
                       [Phase 4: Primary Inference] ➔ [Output Response]
                                  ↓
                       [Phase 5: Async LLM-as-Judge Evaluation]
                                  ↓
                       [Phase 5.5: Telemetry & Billing Log]
                                  ↓
                       [Phase 6: Cache Update]
```

### The Execution Lifecycle

#### Phase 0: Semantic Cache Check
* **Operation:** `QueryCache.get(query, document_scope)`
* **How it works:** The system checks if the exact query has been processed recently under the same patient scope. A deterministic SHA-256 hash is generated from the normalized query string and document scope.
* **Why it matters:** Serving direct cache hits eliminates downstream API costs and reduces latency to milliseconds. Deterministic hashing is safer in zero-tolerance domains than semantic caching, which risks false positive matches on distinct medical terms.

#### Phase 1: Dense Vector Retrieval
* **Operation:** `VectorStore.search(query, k=10)`
* **How it works:** Queries the vector database to retrieve the top 10 most similar small child chunks.
* **Why it matters:** In multi-threaded environments, local library instances (like FAISS) require explicit thread locking to prevent corruption. Distributed servers (such as Qdrant) handle this natively.

#### Phase 2: Relevance Reranking
* **Operation:** `CrossEncoder.rerank(query, candidates)`
* **How it works:** Jointly evaluates query-chunk relevance, keeping only the top 3 highest-scoring candidate chunks.

#### Phase 2.5: Context Expansion
* **Operation:** `parent_store[child.parent_id]`
* **How it works:** Retrieves the larger parent chunks corresponding to the matched child IDs from a key-value store.

#### Phase 2.6: Context Optimization & Token Budgeting
* **Operation:** `trim_to_token_budget(context, max_tokens=6000)`
* **How it works:** Ensures total prompt context fits within optimal limits, prioritizing the highest-relevance chunks.

#### Phase 3: Token Auditing
* **Operation:** `count_tokens(query + context)`
* **How it works:** Audits prompt token size before invocation to track costs and prevent model context overflow.

#### Phase 3.5: Intelligent Model Routing
* **Operation:** `select_model(query, tokens_input)`
* **How it works:** Routes to a faster, low-cost model *only* if the query is short, lacks complex clinical keywords (like *contraindication* or *pathophysiology*), and token count is low. Otherwise, routes to a high-capacity model.
* **Why it matters:** resaves up to 90% in inference costs on simple queries while preserving frontier model power for complex clinical reasoning.

#### Phase 4: Primary Inference
* **Operation:** `LLM.invoke(prompt)`
* **How it works:** Selected LLM processes the prompt and returns the clinical response.

#### Phase 5: Automated Quality Evaluation
* **Operation:** `Judge.evaluate(query, context, response)`
* **How it works:** A separate deterministic judge processes the evaluation in the background, outputting Faithfulness, Answer Relevance, and Context Precision scores.

#### Phase 5.5: Telemetry & Billing Log
* **Operation:** `TokenBudget.record(...)`
* **How it works:** Logs exact token metrics, model types, execution durations, and costs to an audit database.

#### Phase 6: Cache Update
* **Operation:** `QueryCache.set(...)`
* **How it works:** Writes the verified response to the cache registry for future lookup.

---

## Selecting the Right Vector Database

Choosing the appropriate vector database is a critical architectural decision for enterprise clinical RAG systems:

| Vector Database | Architecture | Key Advantages | Major Limitations | Clinical Production Suitability |
| :--- | :--- | :--- | :--- | :--- |
| **FAISS (Facebook)** | Local, In-process library | Ultra-fast local reads, zero network latency. | In-memory only; no native horizontal scaling or metadata filtering; requires manual read/write locking in multi-threaded code. | Excellent for single-node deployments, embedded clinical offline systems, or read-only caches. |
| **ChromaDB** | SQLite/In-process or Client-Server | Easiest setup, rich metadata filtering, active ecosystem. | Scaling limitations at massive query volumes, primarily single-node. | Excellent for clinical prototypes, internal hospital departmental apps, and medium datasets. |
| **Qdrant** | Distributed, Client-Server (Rust) | Rich payload filtering, highly performant, horizontal clustering, cluster safety. | Infrastructure setup overhead. | **Highly recommended** for scalable enterprise healthcare systems requiring strict metadata filtering (e.g., filtering by department or physician). |
| **pgvector (Postgres)** | RAG-extended Relational DB | Leverages mature Postgres ACID transactions, unified storage for patient records and vectors. | Slower index build times, database resources shared between structured queries and vector search. | Ideal for pipelines requiring strict patient data ACID compliance and joining relational charts directly with medical embeddings. |
| **Milvus** | Highly Distributed, Enterprise | Massive scale, sharded ingestion, distributed multi-node architecture. | Extreme operational complexity and heavy resource consumption. | Best suited for country-scale patient databases or massive unified hospital networks. |

---

## The Reality of Operational Complexity and Trade-offs

A highly engineered clinical RAG system looks flawless on paper, but in the real world, **every added architectural layer carries a substantial infrastructure and operational burden.** Engineers must approach these designs with deep pragmatism:

* **Infrastructure Footprint:** Implementing this blueprint requires maintaining a local EasyOCR worker, a distributed vector store (like Qdrant), a high-speed key-value cache (like Redis), an asynchronous task worker (like Celery), a relational EHR connector, a local embedding/reranking host, and a PHI redaction engine. Managing this distributed footprint introduces substantial configuration, logging, and monitoring overhead.
* **Failure Coupling:** Every component added to the query path is a potential point of failure. If the Redis cache fails, the system faces sudden latency spikes. If the asynchronous task queue backs up, evaluation telemetry degrades. Production teams must write resilient fallback code for every single component, ensuring the core RAG system degrades gracefully rather than crashing.
* **The Complexity-Safety Balance:** For standard commercial search or general-purpose assistants, this level of engineering is classic over-design. However, in clinical deployments, this complexity is the non-negotiable cost of patient safety. The extra 400 milliseconds of latency and the operational burden of local secure VPC hosting represent a deliberate trade-off: trading raw speed and simple infrastructure for deterministic clinical reliability.

---

## Conclusion

Naive RAG fails in healthcare at four distinct structural points. Ingestion silently drops non-digital records. Fixed-size chunking severs critical clinical entities and causes embedding signal dilution. Similarity-ordered retrieval buries high-signal context where LLM attention is weakest. Manual evaluation provides no diagnostic metrics until degradation is already severe.

The production design patterns that address these failures are: dual-layer adaptive OCR; parent-child chunking with entity-level retrieval and context-level generation; cross-encoder reranking to enforce relevance ordering before context assembly; and deterministic LLM-as-Judge scoring at `temperature=0` after every generation.

These components form a stateful pipeline where failure at any stage propagates forward. A silently discarded ingestion page produces no child chunk, no parent context, no retrieved evidence, and ultimately a faithfulness score of zero. The observability at the end of the pipeline is only useful when the data integrity at the beginning is guaranteed.

Ultimately, clinical RAG systems do not fail because the LLM is weak. They fail because every retrieval system silently accumulates missing context, degraded evidence, and invisible ingestion errors until the model confidently reasons over absence. Clinical RAG is not a model generation problem; it is a systems engineering problem.

---

### References & Deep Reading

* **Contextual Attention Dynamics:** Liu, N. F., et al. (2023). *Lost in the Middle: How Language Models Use Context*. Stanford University & UC Berkeley.
* **Biomedical Representation Models:** Lee, J., et al. (2020). *BioBERT: a pre-trained biomedical language representation model for biomedical text mining*. Bioinformatics, Oxford Academic.
* **Cross-Encoder Relevance Optimization:** Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. New York University.
* **Evaluating RAG Architectures (Judge Frameworks):** Zheng, L., et al. (2023). *Judging LLM-as-a-Judge on MT-Bench and Chatbot Arena*. UC Berkeley.
* **Retrieval-Augmented Generation Foundational Work:** Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Meta AI.

---

<div class="author-card">
  <h3>Connect & Collaborate</h3>
  <p>Have questions about scaling RAG pipelines in high-stakes clinical domains, or interested in collaborating on robust, production-grade clinical AI applications? Let's connect!</p>
  <div class="author-links">
    <a href="https://www.linkedin.com/in/jha-devesh/" target="_blank" class="author-link-btn linkedin-btn">
      <svg class="author-icon" viewBox="0 0 24 24"><path d="M19 3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h14m-.5 15.5v-5.3a3.26 3.26 0 0 0-3.26-3.26c-.85 0-1.84.52-2.32 1.3v-1.11h-2.79v8.37h2.79v-4.93c0-.77.62-1.4 1.39-1.4a1.4 1.4 0 0 1 1.4 1.4v4.93h2.79M6.88 8.56a1.68 1.68 0 0 0 1.68-1.68c0-.93-.75-1.69-1.68-1.69a1.69 1.69 0 0 0-1.69 1.69c0 .93.76 1.68 1.69 1.68m1.39 9.94v-8.37H5.5v8.37h2.77z"/></svg>
      LinkedIn Profile
    </a>
    <a href="mailto:deveshjnv2002@gmail.com" class="author-link-btn email-btn">
      <svg class="author-icon" viewBox="0 0 24 24"><path d="M20 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 4l-8 5-8-5V6l8 5 8-5v2z"/></svg>
      Email Address
    </a>
  </div>
</div>
