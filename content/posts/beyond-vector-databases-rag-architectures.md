---
title: "Beyond Vector Databases: Exploring Different Types of RAG Architectures"
date: 2026-11-13T12:00:00+05:30
description: "RAG is not just vector search. Six retrieval architectures — vector, keyword, hybrid, graph, agentic, and self-correcting — explained with analogies, diagrams, code, and a decision framework."
summary: "A practical guide to six RAG architectures beyond the standard vector-search pipeline: when each works, when it fails, and how to choose the right one for your problem."
keywords: ["RAG Architectures", "Vector Database", "BM25", "Hybrid RAG", "Graph RAG", "Agentic RAG", "CRAG", "Retrieval-Augmented Generation"]
tags: ["RAG", "LLM Architecture", "NLP", "Information Retrieval"]
TocOpen: true
draft: false
---

> *This article is for educational purposes. It discusses architectural patterns for RAG systems and does not endorse any specific vendor or tool.*

When most developers hear "RAG," they picture one workflow: chunk documents, embed them, store the embeddings in a vector database, retrieve by similarity, and generate an answer. That pipeline appears in nearly every tutorial, and it is a good starting point for many prototypes.

But it is not the only way to build a RAG system, and in some cases it is not the right one. The failure mode is familiar: a legal system looking for exact clause numbers, a clinical assistant looking for drug-interaction paths, or a support bot using identifiers and codes that embeddings do not prioritize. In those cases, a different retrieval strategy may matter more than a different model.

Consider a legal assistant. A lawyer asks: *"Find all clauses that reference Section 12(b)(3) of the Securities Act."* Vector search may return passages about securities regulations and investor protections — semantically relevant, but not the exact clause reference the user needed. The right clause is still in the corpus, but the retrieval method may not surface it efficiently.

Or consider a medical knowledge base. A doctor asks: *"Which of this patient's current medications interact with warfarin?"* Answering this requires connecting patient records, medication lists, and drug-interaction relationships — something a single vector lookup usually does not model well.

These are not edge cases. They are common workloads. The main point is not that vector search is bad; it is that **retrieval strategy is a design choice that should match the problem**, and some real systems need more than a single vector index.

In a [previous post](/posts/why-naive-rag-fails-healthcare/), I explored why naive RAG fails in healthcare and how to fix it. This post zooms out: beyond that single pipeline, what other patterns exist, when does each one work best, and how do you decide?

> **TL;DR: what you'll learn**
>
> | Pattern | What it is | Typical use | Tradeoff |
> | :--- | :--- | :--- | :--- |
> | Vector RAG | Retrieval by embedding similarity | Semantic questions, paraphrase matching | Weak on exact identifiers |
> | Keyword RAG (BM25) | Retrieval by term overlap | Exact terms, codes, IDs, entity names | Weak on paraphrase and concept matching |
> | Hybrid RAG | Combine vector and keyword retrieval | General production search | More infrastructure complexity |
> | Graph RAG | Retrieve by traversing relationships | Multi-hop and entity-relationship questions | More engineering and extraction complexity |
> | Agentic RAG | LLM chooses tools and search paths | Multi-source, open-ended research | Higher latency and cost |
> | Self-Correcting RAG | Evaluate and retry/refuse before answering | High-stakes domains | Adds extra evaluation loop |
>
> These are not all the same kind of thing. Some are retrieval methods, some are orchestration patterns, and some are quality-control layers. The point is not to memorize them as a rigid taxonomy, but to understand the design choices they represent.

---

## Two Ways to Think About RAG

Before diving into each pattern, it helps to separate two design decisions that often get conflated:

1. **How do you search?** — This is the *retrieval method*: vector similarity, keyword matching, SQL queries, graph traversal, or a combination of them.
2. **How do you coordinate?** — This is the *orchestration pattern*: do you search once and generate, evaluate the results and retry, or let an agent choose tools and plans across multiple sources?

These axes are independent. You can combine a vector retrieval method with self-correcting orchestration, or a hybrid retrieval method with an agentic coordinator. The six patterns below sit at different points in this space:

```text
                         How do you search?
                 ┌────────────────────────────────────┐
                 │  Vector    Keyword   Hybrid   Graph │
   How do   ─────┼────────────────────────────────────┤
   you      One  │  Vector    BM25     Hybrid   Graph │
   coordinate?   │  RAG       RAG      RAG      RAG   │
            ─────┼────────────────────────────────────┤
            Check│  Self-correcting RAG                │
            first│  (wraps around any search method)   │
            ─────┼────────────────────────────────────┤
            Plan │  Agentic RAG                        │
            &    │  (picks which search to use)        │
            adapt│                                     │
                 └────────────────────────────────────┘
```

The top row is about **what kind of search** you use. The bottom two rows are about **how the system chooses and verifies retrieval**. This is not meant to be a formal taxonomy; it is a map of the design space.

The practical lesson: do not pick a pattern just because it sounds advanced. Pick it because it solves the retrieval problem you actually have. Let's start from the simplest and build up.

---

## A Quick RAG Refresher

If you already know how RAG works, skip to the next section. If not, here is the idea in four sentences.

A language model can only answer from what it has seen during training or what you put in its prompt. **Retrieval-Augmented Generation** puts relevant documents into the prompt so the model can answer questions about your private data (Lewis et al., 2020). The standard pipeline has two phases: an **offline** phase (process and index your documents) and an **online** phase (receive a question, retrieve relevant chunks, and generate an answer). Everything below is a variation on *how* those two phases work.

```text
Offline:   Documents → Split into chunks → Convert to embeddings → Store in index

Online:    Question → Search the index → Retrieve top chunks → Pass to LLM → Answer
```

This default pipeline uses **vector search** (embedding similarity) as its retrieval method. The six architectures below each change either the retrieval method, the orchestration around it, or both.

---

## 1. Vector RAG — The Semantic Baseline

**Analogy:** Describing a book to a librarian who understands meaning. You say *"I need something about managing high blood pressure in elderly patients,"* and the librarian finds relevant books even if none use those exact words.

### How it works

Each document chunk is converted into a dense numerical vector (an "embedding") using a model like `all-MiniLM-L6-v2`. These vectors capture semantic meaning: chunks about similar topics land close together in vector space. At query time, the question is also embedded, and a vector database (FAISS, Pinecone, Weaviate, Qdrant, or Chroma) returns the nearest chunks by cosine similarity.

```text
Question: "What causes heart failure?"
                ↓
        Embed the question
                ↓
  Find nearest vectors in the index
                ↓
  Top chunks (even if they say "cardiac insufficiency"
  instead of "heart failure") → LLM → Answer
```

### When it works

- **Semantic questions** where the user's wording varies: "heart attack" matches "myocardial infarction."
- Unstructured text: articles, reports, documentation, PDFs.
- Moderate-sized corpora where embedding quality stays high.

### When it fails

- **Exact terms.** ICD codes (`E11.9`), drug names, section references, phone numbers — embeddings compress these into the same region of vector space as similar-looking but factually different terms. `E11.9` (Type 2 diabetes) and `E10.9` (Type 1 diabetes) may end up with nearly identical embeddings.
- **Long documents** where a single chunk loses context. (Parent-child chunking addresses this — I covered it in [the previous post](/posts/why-naive-rag-fails-healthcare/).)
- Queries that depend on exact word matching, not semantic similarity.

> **Remember:** Vector retrieval is strongest when semantic similarity matters. It is not a substitute for exact-match lookup when the task depends on IDs, codes, clause numbers, or product names. In those cases, lexical retrieval or metadata filtering often performs better.

---

## 2. Keyword RAG (BM25) — Purpose-Built for Exact Matches

**Analogy:** Looking up a term in the index at the back of a textbook. You find exactly "Section 12(b)(3)" on page 47. Fast, precise, and reliable — but it will not help if the passage uses different words for the same concept.

### How it works

BM25 (Best Matching 25) is a scoring function from the 1990s that ranks documents by how well they match the query's keywords, weighted by term frequency and document length (Robertson & Zaragoza, 2009). No embeddings, no neural network, no vector database — just word matching with smart weighting. Elasticsearch and OpenSearch use BM25 by default.

**This is a fully vector-free RAG architecture.** You do not need an embedding model or a vector database to build it. An inverted index and a language model are enough.

```text
Question: "metformin dosage for E11.9"
                ↓
         Tokenize into keywords
                ↓
  Look up "metformin," "dosage," "E11.9" in inverted index
                ↓
  Score each document by keyword overlap (BM25) → Top-k → LLM → Answer
```

An **inverted index** maps every word to the list of documents that contain it — like the index at the back of a book, but for the entire corpus. Lookup is nearly instant, even on millions of documents.

### When it works

- **Exact identifiers:** drug names, legal clause numbers, error codes, product SKUs, ICD codes.
- Terminology-heavy domains where the same term always means the same thing.
- Very large corpora — inverted indices are among the fastest search structures in computer science.
- When you want a simple, low-cost setup without GPU-heavy embedding models.

### When it fails

- **Synonyms and paraphrases.** "Heart attack" will not match a document that only says "myocardial infarction."
- Semantic questions where the user's wording differs from the source text.

> **Remember:** BM25 is strongest when the query is built around exact terms, IDs, or names. It is much weaker when the task depends on synonyms, paraphrase, or latent concept similarity. In the real world, most production systems use it as one input to a larger retrieval pipeline, not as a standalone answer to every question.

---

## 3. Hybrid RAG — Combining Semantic and Lexical Retrieval

**Analogy:** Using *both* the librarian (who understands meaning) *and* the textbook index (which finds exact terms), then merging their recommendations into one ranked list.

### How it works

Run vector search and BM25 in parallel on the same query, then combine the two ranked lists using a fusion strategy. The most common is **Reciprocal Rank Fusion (RRF):**

```text
Question ──┬──→ Vector search → Ranked list A ──┐
           │                                     ├──→ RRF merge → Final ranked list → LLM → Answer
           └──→ BM25 search   → Ranked list B ──┘
```

RRF scores each document by its reciprocal rank in each list and sums the scores. A document ranked #1 in one list and #5 in the other scores higher than a document ranked #3 in both. It is simple, robust, and widely used, but it is not automatically the best fusion method for every workload (Cormack et al., 2009).

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked lists using RRF.
    k controls how much lower ranks are discounted (default 60)."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)

# Example: merge vector search results and BM25 results
final_ranking = reciprocal_rank_fusion([vector_results, bm25_results])
```

### When it works

- **Many production RAG systems converge on hybrid search**, because real-world queries mix semantic intent with exact terms: *"What is the recommended dosage of metformin for Type 2 diabetes (ICD E11)?"*
- Any domain where users sometimes use jargon and sometimes use plain language.
- When you need both recall (finding relevant content) and precision (finding exact matches).

### When it fails

- Still **single-hop**: one retrieval step, one set of results. If the answer requires chaining multiple facts together, hybrid search alone will not solve it.
- Adds infrastructure complexity: you now need both a vector index and a keyword index.

### What I'd add to my project

My Healthcare RAG system currently uses vector search with a cross-encoder reranker. Adding BM25 for hybrid retrieval is an improvement I'd consider — it would help catch the exact drug names and ICD codes that embeddings sometimes miss.

> **Remember:** Hybrid search combines the strengths of vector and keyword retrieval. It is a strong default when your workload mixes semantic intent with exact names, IDs, or terminology, but it is still a single-hop retrieval step. It does not magically solve multi-step or relationship-heavy questions.

---

## 4. Graph RAG — When Relationships Matter

**Analogy:** A detective's evidence board — pins for people, places, and events, connected by strings. You do not just search for a keyword; you follow connections. *"Who was at the scene AND has a prior record AND is connected to the suspect?"*

### How it works

Instead of (or in addition to) embedding chunks, you build a **knowledge graph** where entities are nodes and their relationships are edges. Retrieval traverses the graph to find connected information that no single chunk contains.

```text
Offline:   Documents → Extract entities & relationships → Build knowledge graph

Online:    Question → Identify entities in the question
                         ↓
             Traverse graph from those entities
                         ↓
             Collect connected subgraph → LLM → Answer
```

A question like *"Which of this patient's medications interact with warfarin?"* needs three steps:

1. Find the patient node and their medication edges.
2. For each medication, look up drug-interaction edges.
3. Filter for interactions involving warfarin.

No single vector lookup can do this. The graph connects the dots.

### When it works

- **Multi-hop questions:** *"Which suppliers of company X also supply its competitor Y?"*
- **Relationship-heavy domains:** medicine (drug interactions, symptom–disease links), law (case precedent chains), fraud detection (transaction networks), supply chain analysis.
- When you need **explainable** retrieval — graph traversal paths are human-readable, unlike vector similarity scores.

### When it fails

- **Graph construction is expensive and error-prone.** Extracting entities and relationships from unstructured text is itself an LLM task, and extraction errors become retrieval errors. Microsoft's GraphRAG work (2024) demonstrated the potential of this approach but also highlighted the cost and complexity.
- **Simple factual questions** do not need a graph. "What is the dosage of aspirin?" is faster and cheaper with a vector or keyword lookup.
- Adds operational complexity: you need a graph database (Neo4j, Amazon Neptune) alongside your other indices.

> **Remember:** Graph RAG shines when the *connections between facts* matter more than the facts themselves. If your questions are about individual facts, a graph adds complexity you may not need.

---

## 5. Agentic RAG — Let the System Decide

**Analogy:** Instead of you choosing which library to visit, you hire a research assistant. You tell them the question, and they decide: *"I'll check the medical database first, then cross-reference with the drug interaction list, and if I still don't have enough, I'll search the clinical guidelines."* They plan, execute, and evaluate — on their own.

### How it works

An LLM-powered **agent** receives the question and has access to multiple retrieval **tools** — vector search, SQL queries, graph lookups, web search, APIs. The agent decides which tool to call, what query to send, whether the result is good enough, and whether to try another tool.

```text
User question → Agent (LLM with tools)
                   │
                   ├──→ Tool 1: Vector search on clinical docs → results
                   ├──→ Tool 2: SQL query on patient database → results
                   ├──→ Tool 3: Graph lookup for drug interactions → results
                   └──→ Tool 4: Web search for latest guidelines → results
                          ↓
                 Agent evaluates all results
                          ↓
                 Enough? ──→ Yes → Generate answer with citations
                          └─→ No  → Reformulate and call another tool
```

Frameworks like LangChain Agents, LlamaIndex Agents, and CrewAI provide abstractions for this pattern, but the core idea is straightforward: the LLM acts as a **router and planner**, not just a generator.

### When it works

- Questions that **span multiple data sources:** *"Based on this patient's records AND the latest WHO guidelines, what treatment options exist?"*
- **Open-ended research** where the right data source is not known in advance.
- Complex workflows where retrieval needs to be iterative and adaptive.

### When it fails

- **Latency.** Every tool call involves an LLM inference step plus a retrieval call. A simple question that needs one lookup can take 3–5× longer through an agent.
- **Cost.** Multiple LLM calls per question add up, especially with larger models.
- **Reliability.** The agent can choose the wrong tool, form a bad query, or loop without converging. **Guardrails are essential:** maximum iterations, timeouts, and a fallback to a simpler pipeline.
- **Simple questions.** *"What is the capital of France?"* does not need an agent — it needs one fast lookup.

> **Remember:** Agentic RAG is powerful but adds latency, cost, and failure modes. Use it when the question genuinely requires multi-step reasoning or multi-source retrieval, not as a default.

---

## 6. Self-Correcting RAG (CRAG) — Check Before You Answer

**Analogy:** A journalist who fact-checks every article before publishing. Write the draft, verify each claim against the sources. If a claim is not supported, rewrite it or remove it. If nothing in the sources supports the answer at all, say *"I don't have enough information"* instead of guessing.

### How it works

After the initial retrieval, the system **evaluates** whether the retrieved documents actually support an answer. Depending on the evaluation, it takes one of three paths (Yan et al., 2024):

```text
User question → Retrieve → Evaluate relevance of retrieved chunks
                               │
                               ├──→ Relevant     → Generate answer → Check faithfulness → Return
                               │
                               ├──→ Ambiguous    → Reformulate query → Retrieve again → Re-evaluate
                               │
                               └──→ Not relevant  → Try a different strategy OR refuse
```

The relevance check can be a **cross-encoder reranker** (fast, lightweight), an **LLM judge** (more nuanced, slower), or both in sequence.

This is not a standalone retrieval method — it is a **quality layer** that wraps around *any* retrieval method. You can self-correct a vector RAG system, a hybrid system, or even an agentic one.

### When it works

- **High-stakes domains** where a wrong answer is worse than no answer: healthcare, legal, finance.
- When your knowledge base is **incomplete** — the system can recognize gaps instead of hallucinating to fill them.
- As a wrapper around any other architecture in this post.

### When it fails

- **Latency.** Every correction loop adds a round trip. In the worst case, the system retries 2–3 times before answering or refusing.
- **Over-refusal.** A conservative relevance threshold can make the system refuse too often, frustrating users who asked perfectly reasonable questions. Threshold tuning is an ongoing process.
- Requires real user queries to calibrate — synthetic test sets are usually not enough.

### What I built

My Healthcare RAG system uses a version of this: a cross-encoder reranker that rejects chunks below a relevance threshold, and an LLM judge that scores faithfulness after generation. If I were rebuilding it, I would add the retry loop — reformulate the query and search again before refusing.

> **Remember:** Self-correcting RAG trades speed for safety. In healthcare and legal domains, that trade-off is usually worth it. In a chatbot for a recipe site, it is probably not.

---

## Putting It Together: A Combined Reference Architecture

Each architecture above solves one problem. A production system often combines several of them. Here is one reasonable way to put the pieces together — not the only way, but one that covers the most common failure modes:

```text
[User question]
       ↓
[0. Route: is the question simple or complex?]
       ├──→ Simple (single fact, clear terms)
       │         ↓
       │    [1. Hybrid search: vector + BM25 in parallel]
       │         ↓
       │    [2. RRF merge → Cross-encoder rerank → Top 3]
       │         ↓
       │    [3. Generate answer with citations]
       │         ↓
       │    [4. Self-correction: evaluate faithfulness]
       │         ├──→ Passed → Return answer
       │         └──→ Failed → Refuse or retry with reformulated query
       │
       └──→ Complex (multi-source, multi-step, or relationship question)
                 ↓
            [1. Agent selects tools based on the question]
                 ├──→ Hybrid search on document corpus
                 ├──→ Graph traversal for entity relationships
                 ├──→ SQL query for structured data
                 └──→ Web search for recent information
                 ↓
            [2. Agent evaluates: enough evidence?]
                 ├──→ No  → Call another tool or reformulate
                 └──→ Yes → Merge results, rerank, apply token budget
                              ↓
                         [3. Generate answer with citations]
                              ↓
                         [4. Self-correction: evaluate faithfulness]
                              ├──→ Passed → Return answer
                              └──→ Failed → Refuse
                                       ↓
                                  [5. Cache grounded answers, log usage and cost]
```

The architectures from this post map onto this flow:
- **Step 0** is a simple router (question length, keywords, number of entities) — not every question needs the agentic path.
- **Steps 1–2 on the left** are **Hybrid RAG** with self-correction.
- **Steps 1–2 on the right** are **Agentic RAG**, which can call **Graph RAG**, **SQL**, or **Hybrid search** as tools.
- **Step 4 on both paths** is **Self-Correcting RAG (CRAG)**.
- **Keyword RAG (BM25)** is part of every hybrid search call.
- **Vector RAG** is the other half of every hybrid search call.
- [Parent-child chunking and OCR](/posts/why-naive-rag-fails-healthcare/) happened at ingestion time, before any question is asked.

Not every system needs every layer. Start with the simplest path that meets your retrieval requirements, measure your failures, and add complexity only when the evidence justifies it. Each layer adds latency, cost, and operational complexity.

---

## How to Evaluate a RAG Architecture

Choosing a pattern without measurement is guesswork. A practical evaluation should track more than just the final answer correctness. For each architecture, measure:

- **Retrieval quality:** recall@k, nDCG, MRR, and whether the relevant document appears in the top results.
- **Answer quality:** correctness, citation support, hallucination rate, and groundedness.
- **Failure modes:** exact-term misses, multi-hop failures, over-refusal, bad tool selection, or looping agents.
- **System cost:** latency, token usage, compute cost, and database/indexing overhead.
- **Operational reliability:** index freshness, maintenance burden, and failure recovery.

A simple evaluation matrix looks like this:

| Pattern | Best evaluated on | Main failure to watch for | Typical cost profile |
| :--- | :--- | :--- | :--- |
| Vector RAG | Semantic paraphrase tasks | Exact identifiers and rare terms | Low to moderate |
| Keyword RAG | Exact match, code lookup, legal/medical identifiers | Synonyms and concept matching | Low |
| Hybrid RAG | Mixed query workloads | Still single-hop; may still miss graph tasks | Moderate |
| Graph RAG | Multi-hop and entity-relationship tasks | Extraction errors and graph maintenance | Moderate to high |
| Agentic RAG | Multi-source or tool-heavy tasks | Bad routing, loops, latency spikes | High |
| Self-Correcting RAG | High-stakes, low-tolerance tasks | Over-refusal and extra latency | Moderate to high |

For production systems, start with a benchmark built from your real user queries, not just synthetic examples. Label retrieval hits, answer correctness, and confidence failures. If a pattern fails mostly on exact terms, fix that with BM25 or metadata filters before building a more complex graph or agentic layer.

---

## How to Choose: A Decision Framework

Do not pick an architecture because it sounds impressive. Pick it because it solves the specific retrieval problem you have.

{{< rag-picker >}}

| Your situation | Start with | Why |
| :--- | :--- | :--- |
| Unstructured docs, semantic questions | **Hybrid RAG** (vector + BM25) | Covers both meaning and exact terms |
| Structured data in a database | **Text-to-SQL** | Query the data directly — no embeddings needed |
| Questions about relationships between entities | **Graph RAG** | Follow connections, not just match content |
| Questions spanning multiple data sources | **Agentic RAG** | Let the agent decide where to look |
| High-stakes domain (healthcare, legal, finance) | **Hybrid + Self-Correcting** | Verify before answering; refuse if unsure |
| Simple FAQ or documentation lookup | **Vector or Keyword RAG** | Do not over-engineer what a simple search solves |

### They are not mutually exclusive

The reference architecture above combines several of them. The common layering pattern:

- **Hybrid retrieval** (vector + BM25) as the search layer.
- **Hierarchical chunking** (parent-child) for long documents.
- **Self-correction** (reranker + judge + refusal) as the quality layer.
- **Agentic orchestration** only when questions need multiple sources or multiple steps.

My Healthcare RAG system already combines three of these: vector retrieval, hierarchical chunking, and self-correction. Adding BM25 for hybrid search, and an agentic layer for multi-document queries, would be the natural next steps.

---

## Other Approaches Worth Knowing

This post covered six architectures, but the RAG design space is broader. Other patterns that are worth exploring but deserve their own detailed treatment:

- **Contextual RAG** (Anthropic, 2024) — prepend document-level context to each chunk before embedding, so the embeddings capture where each chunk fits in the broader document.
- **Adaptive RAG** — dynamically choose how deep to search (single retrieval vs. iterative) based on question complexity, rather than always using the same pipeline.
- **Multi-modal RAG** — retrieve and reason over images, tables, and text together, not just text chunks.
- **Modular RAG** — treat each pipeline step (chunking, retrieval, reranking, generation) as a swappable module, making it easier to experiment and improve one step at a time.

Each of these could fill its own article. The six architectures in this post cover the most common design decisions you will face when building a RAG system from scratch.

---

## What I'd Build Differently Today

Having built a single-shot hierarchical RAG system for healthcare ([source code](https://github.com/DeveshNathJha/Healthcare-Rag)), here is what I would change:

1. **Add BM25 for hybrid search.** Drug names like "metformin" and ICD codes like "E11.9" are exact identifiers. Vector search sometimes misses them; BM25 would not.
2. **Add a retry loop before refusing.** When the reranker scores everything below threshold, reformulate the query (simplify it, extract keywords) and search again before giving up.
3. **Consider a lightweight graph for drug interactions.** Some questions (*"Does this medication interact with the patient's other prescriptions?"*) are inherently multi-hop. A small graph overlay on top of the vector index could handle these without rebuilding the whole system.
4. **Keep single-shot for simple questions.** Not every question needs an agent. Route simple factual lookups to a fast hybrid search, and only invoke the agent for complex, multi-step queries.

The takeaway: **start simple, measure where your system fails, and add complexity only where those failures demand it.** Every architecture layer you add increases latency, cost, and the surface area for bugs.

---

## Conclusion

RAG is not one architecture. It is a design space with many retrieval methods — some using vectors, some not — and orchestration patterns that can be mixed and matched.

The most common mistake is treating "RAG" as a single pattern — chunk, embed, search, generate — and being surprised when it fails on exact terms, connected data, or multi-source questions. The second most common mistake is the opposite: reaching for an agentic graph RAG system when a simpler approach would have worked.

**Start with hybrid search. Add self-correction if accuracy matters. Add graph or agentic retrieval only when your questions genuinely need them. Measure before you add.**

The [first post](/posts/why-naive-rag-fails-healthcare/) in this series covered why naive RAG fails in healthcare. This post explored what other architectures exist and when to use them. The thread across both: **most RAG quality problems trace back to retrieval choices, not model choices.**

---

### Further Reading

* Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
* Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval.
* Cormack, G., Clarke, C., & Buettcher, S. (2009). *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*. SIGIR 2009.
* Microsoft Research (2024). *GraphRAG: Unlocking LLM Discovery on Narrative Private Data*.
* Yan, S., et al. (2024). *Corrective Retrieval-Augmented Generation (CRAG)*. arXiv preprint.
* Gao, Y., et al. (2024). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv preprint.
* Anthropic (2024). *Contextual Retrieval*. Blog post.

---

## Let's Discuss

Have a question about choosing the right RAG architecture, or building one of these in practice? Let's connect:

<div class="author-card">
  <h3>Connect & Collaborate</h3>
  <p>I write about RAG systems, retrieval quality, and production AI engineering. If you're building a RAG system and want to compare notes, I'd love to hear from you.</p>
  <div class="author-links">
    <a href="https://www.reddit.com/user/JhaDevesh/" target="_blank" class="author-link-btn reddit-btn">
      <svg class="author-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.703zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.197-2.512-.73a.326.326 0 0 0-.232-.095z"/></svg>
      Reddit Profile
    </a>
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
