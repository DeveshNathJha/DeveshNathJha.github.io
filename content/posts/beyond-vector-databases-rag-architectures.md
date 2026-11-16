---
title: "Beyond Vector Databases: Exploring Different Types of RAG Architectures"
date: 2026-11-13T12:00:00+05:30
lastmod: 2026-11-16T12:00:00+05:30
description: "A practical map of RAG architectures beyond plain vector search: retrieval methods, orchestration patterns and quality-control layers, with small reproducible experiments and a decision framework."
summary: "RAG is more than vector search. A practical map of the main design choices (how you search, how you coordinate, how you verify), what each is good and bad at, and three small experiments that show where the common advice breaks."
keywords: ["RAG Architectures", "Vector Database", "BM25", "Hybrid RAG", "Graph RAG", "Agentic RAG", "Corrective RAG", "Reranking", "Retrieval-Augmented Generation"]
tags: ["RAG", "LLM Architecture", "NLP", "Information Retrieval"]
TocOpen: true
draft: false
---

> *This article is for educational purposes. It describes common design patterns for RAG systems and does not endorse any specific vendor or tool. The experiments are small and illustrative, not benchmarks.*

When most developers hear "RAG", they picture one workflow: split documents into chunks, turn the chunks into embeddings, store them in a vector database, retrieve the most similar chunks, and ask a language model to answer. That pipeline appears in most tutorials, and it is a good starting point.

It is not the only way to build a RAG system, though, and for some problems it is not the best one. Two examples:

* A lawyer asks: *"Find all clauses that reference Section 12(b)(3)."* Vector search may return passages that are *about* similar regulations, while the clause that literally mentions "12(b)(3)" ranks lower. The user needed an exact reference, not a similar topic.
* A clinician asks: *"Which of this patient's current medications interact with warfarin?"* The answer has to combine the patient's medication list with drug-interaction knowledge. That is a relationship question across sources, and in practice it is often better served by structured data (a medication table, a curated interaction database) than by searching text at all.

The point is not that vector search is bad. It is that **retrieval is a design choice**, and different questions need different choices. This post maps out those choices and how to decide between them, with three small experiments that test some common claims along the way. (Ingestion, chunking and evaluation are covered in more depth in an earlier post, [Why Naive RAG Fails in Healthcare](/posts/why-naive-rag-fails-healthcare/).)

> **TL;DR**
>
> | Layer | The question it answers | Options covered here |
> | :--- | :--- | :--- |
> | **Retrieval** | How do we find candidate information? | Vector, keyword (BM25), hybrid, metadata filters and SQL, graph |
> | **Orchestration** | How many searches, and who decides? | Single-shot, adaptive, agentic |
> | **Quality control** | How do we check before answering? | Reranking with thresholds, corrective retrieval, answer verification |
>
> These layers combine: a system can use hybrid retrieval, an agent to choose tools, and a reranker plus a verifier on top. "Graph RAG vs. agentic RAG" is usually not an either-or choice.

---

## A Map of the Design Space

Many articles list "types of RAG" side by side: vector RAG, hybrid RAG, graph RAG, agentic RAG, corrective RAG. That list is useful, but it mixes different kinds of decisions. It helps to sort them into three layers:

```text
                        +--------------------------------------------------+
  Quality control       |  reranking + thresholds, corrective retrieval,   |
  "is this good         |  answer verification, refusal                    |
   enough to answer?"   +--------------------------------------------------+
                                            |
                        +--------------------------------------------------+
  Orchestration         |  single-shot  ->  adaptive  ->  agentic          |
  "how many searches,   |  (one search)    (decide how    (an LLM picks    |
   and who decides?"    |                   much to do)    tools and steps)|
                        +--------------------------------------------------+
                                            |
                        +--------------------------------------------------+
  Retrieval             |  vector | keyword (BM25) | hybrid | metadata     |
  "how do we find       |  filters | SQL | graph                           |
   information?"        +--------------------------------------------------+
```

With this map, many design questions become easier to frame. "Should I use graph RAG or agentic RAG?" turns into two separate questions: does my data need graph *retrieval*, and do my questions need agentic *orchestration*? An agent can use graph retrieval as one of its tools.

The rest of the post walks through the layers from the bottom up.

### A quick RAG refresher

A language model can only answer from what it learned in training or what is in its prompt. **Retrieval-Augmented Generation** (Lewis et al., 2020) finds relevant passages from your own data and puts them in the prompt, so the model can answer from them and cite them. A RAG system has an **offline** part (process and index documents) and an **online** part (receive a question, retrieve, generate):

```text
Offline:  documents -> split into chunks -> index (embeddings and/or keywords)
Online:   question -> retrieve top chunks -> prompt with the chunks -> LLM -> answer with citations
```

Everything below changes *how* the retrieve step works, *how many times* it runs, or *what checks* happen around it.

---

## Layer 1: Retrieval

### Vector retrieval: matching by meaning

**How it works.** An embedding model turns each chunk into a vector (a list of numbers, for example 384 of them for `all-MiniLM-L6-v2`). Texts with similar meaning tend to get vectors that point in similar directions. At query time, the question is embedded the same way, and a vector index (FAISS, Chroma, Qdrant, pgvector and others) returns the chunks with the highest cosine similarity.

**Good at:** paraphrases and concept matching. A question about "heart attack" can find a passage that only says "myocardial infarction".

**Weak at:** exact identifiers. Embeddings represent overall meaning, not exact strings, so two codes that look alike but mean different things can land close together.

**A small experiment.** How close do two different diagnosis codes get? I measured cosine similarity (1.0 means identical direction) with two open-source embedding models, `all-MiniLM-L6-v2` and `bge-small-en-v1.5`. Guess first, then see the results:

{{< similarity-guess >}}

In this check, two *different* diabetes codes were more similar to each other than a pair of *true synonyms*, in both models. Two different drugs with similar spelling also scored higher than the synonym pair in the smaller model. This is only two models and a handful of pairs, and scores are model-specific, but it suggests why exact identifiers often need help from other retrieval methods. (The code to reproduce this is at the end of the post.)

### Keyword retrieval (BM25): matching by words

**How it works.** BM25 (Robertson & Zaragoza, 2009) scores a document by the query words it contains. Three ideas drive the score:

* **Term frequency:** a word that appears more often in a document counts more, but with diminishing returns.
* **Inverse document frequency:** rare words (like `E11.9`) count much more than common words (like "patient").
* **Length normalization:** long documents do not win just because they contain more words.

It runs on an **inverted index**, a map from each word to the documents that contain it, much like the index at the back of a book. Elasticsearch and OpenSearch use BM25 as their default ranking function. No embedding model or vector database is needed, so a keyword-only RAG system is entirely possible.

**Good at:** rare, exact terms: drug names, clause numbers, error codes, product IDs.

**Weak at:** synonyms and paraphrases. "Heart attack" will not match a passage that only says "myocardial infarction".

**A common misconception: BM25 does not guarantee exact matches.** It ranks *tokens* (the pieces the text is split into), so everything depends on how text and queries are tokenized. A small experiment with the `rank_bm25` library and a document containing "Patient diagnosed with E11.9, started on metformin":

| Setup | Query | Score of the right document |
| :--- | :--- | ---: |
| Whitespace tokenizer | `E11.9` | **0.000** (the document's token is `e11.9,` with the comma attached) |
| Tokenizer that keeps codes together and drops punctuation | `E11.9` | 0.463 |
| Same tokenizer | `E119` (written without the dot) | **0.000** |

{{< token-playground >}}

So for identifiers, BM25 works best together with **normalization** (one standard form for each code), an **exact-match field** or metadata filter for known ID formats, and tests that cover the formats users actually type.

### Hybrid retrieval: combining both

**How it works.** Run vector search and BM25 on the same question, then merge the two ranked lists. A widely used merge method is **Reciprocal Rank Fusion (RRF)** (Cormack et al., 2009): each document gets `1 / (k + rank)` from every list it appears in, and the totals are summed.

```text
question --+--> vector search --> ranked list A --+
           |                                      +--> RRF merge --> final list --> (rerank) --> LLM
           +--> BM25 search   --> ranked list B --+
```

```python
def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge several ranked lists of document ids. k dampens the influence of top ranks."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
```

RRF only uses ranks, not raw scores, which is a key advantage: vector similarities and BM25 scores live on completely different scales, and RRF does not need to compare them. With `k = 60`, a document ranked 1st in one list and 5th in the other scores 1/61 + 1/65 ≈ 0.03178, while a document ranked 3rd in both scores 2/63 ≈ 0.03175. The difference is tiny: RRF mostly rewards documents that appear near the top of *both* lists.

{{< rrf-playground >}}

RRF is easy to set up, but not completely tuning-free. The constant `k`, how many candidates each retriever returns before merging, and optional per-retriever weights all affect results.

**Good at:** mixed workloads, where some questions are about meaning and others contain exact terms. Several search engines and vector databases now offer hybrid search as a built-in feature.

**Trade-offs:** two indexes to build and keep in sync, more moving parts, and more to evaluate. Hybrid is not automatically better than either method alone; whether it helps depends on your questions, which is why measuring matters (see [How to evaluate](#how-to-evaluate-a-rag-system)).

### Metadata filters and SQL: when the answer is structured

Two retrieval methods are easy to overlook because they are not "AI":

* **Metadata filtering** narrows the search space before or during retrieval: by document type, date, department, source or the user's permissions. In real systems, this can matter as much as the choice between vector and keyword search. A correct answer from a document the user is not allowed to see is still a wrong answer.
* **SQL (Text-to-SQL)**: when the answer lives in tables ("How many patients were readmitted last month?"), an LLM can translate the question into a SQL query and the system runs it. This avoids embedding table rows at all. Doing it safely needs a read-only database user, validation of the generated query, limits on which tables are visible, and the same permission checks as the rest of the application.

### Graph retrieval: when relationships matter

**How it works.** Information is stored as a **knowledge graph**: entities (patients, drugs, companies, cases) are nodes, and relationships (*takes*, *interacts with*, *supplies*, *cites*) are edges. Retrieval follows edges to collect connected facts that no single chunk contains.

"Graph RAG" is used for several different things, and it helps to keep them apart:

* **Querying an existing knowledge graph.** The graph is curated (for example, a drug-interaction database). The system translates the question into a graph query, and the LLM explains the results. Much of the reliability comes from the curated data.
* **Building a graph from documents with an LLM.** An LLM extracts entities and relationships from text, and retrieval traverses the result. This works on unstructured data, but extraction errors become retrieval errors.
* **Graph-based summarization.** Microsoft's GraphRAG work (Edge et al., 2024) extracts a graph, groups it into communities of closely connected entities, writes a summary for each community, and uses those summaries to answer broad "global" questions, such as the main themes across a whole document collection, which the paper reports top-k chunk retrieval handles poorly.

**Good at:** multi-hop questions ("Which suppliers of company X also supply company Y?") and domains where connections matter more than individual facts. Traversal paths can also be shown to users as an explanation.

**Hard parts:** recognizing that "Metformin", "metformin HCl" and "Glucophage" are the same entity; resolving conflicting relationships; keeping each edge linked to its source document; and updating the graph when documents change. For simple factual questions, a graph adds cost without adding much.

> **Retrieval in one line:** match the retrieval method to how your questions are phrased (meaning, exact terms, structure or relationships), and use metadata filters everywhere.

{{< check q="Your documents write a diagnosis code as `E11.9`, but users often type `E11-9`. Which change is most likely to help?" options="Switch to a larger embedding model|Normalize identifiers to one form, both when indexing and when searching|Retrieve the top 50 chunks instead of the top 10" answer="2" explain="Normalization turns both spellings into the same token, so keyword search can match them. A larger embedding model may still treat similar-looking codes as close, and retrieving more chunks mostly adds noise." >}}

{{< check q="With RRF and k = 60, which document ranks higher: one ranked 1st in one list and 5th in the other, or one ranked 3rd in both?" options="1st and 5th, by a large margin|1st and 5th, but only by a tiny margin|3rd in both" answer="2" explain="1/61 + 1/65 is about 0.03178, and 2/63 is about 0.03175. RRF mainly rewards documents near the top of both lists. Try it in the playground above." >}}

---

## Layer 2: Orchestration

### Single-shot, adaptive and agentic

Orchestration decides how many retrieval steps run and who decides them:

* **Single-shot:** retrieve once, then answer. Predictable, cheap and easy to test, which makes it a sensible place to start.
* **Adaptive:** choose how much retrieval to do per question: no retrieval, one search, or several steps. Adaptive-RAG (Jeong et al., 2024) trains a small classifier to predict question complexity and pick the strategy.
* **Agentic:** an LLM agent gets a set of tools (vector search, SQL, graph queries, web search, APIs) and decides which to call, with what query, and whether it has enough evidence to answer.

```text
question -> agent (LLM with tools)
               |-- tool: hybrid search over documents
               |-- tool: SQL query over structured data
               |-- tool: graph lookup for relationships
               |-- tool: web search for recent information
               v
            enough evidence? -- no --> call another tool or rewrite the query (within a step budget)
               | yes
               v
            answer with citations
```

**Good at:** questions that span several sources or need several dependent steps, where the right path is not known in advance.

**Hard parts.** Agentic systems add failure modes that single-shot systems do not have:

* **Wrong tool or wrong query.** The agent is an LLM making probabilistic decisions; it can pick a poor tool or phrase a poor search.
* **Loops and runaway cost.** Without a maximum number of steps and a time budget, an agent can keep searching. Every step adds at least one LLM call, so latency and cost grow with the number of steps.
* **Growing context.** Each tool result is added to the prompt, which can crowd out the evidence that matters.
* **Prompt injection.** Retrieved documents and web pages can contain instructions ("ignore previous instructions and..."). Retrieved text must be treated as data, never as instructions to follow (Greshake et al., 2023).
* **Harder to test and debug.** The same question can take different paths on different runs, so step-by-step tracing is essential.

A practical middle ground is a **fixed workflow with a router**: a few predefined paths (for example, "document search", "SQL", "both") and a classifier or simple rules that pick one. This is easier to test than a free-form agent, and for many applications it may be enough.

> **Orchestration in one line:** add decision-making only when questions really need several sources or steps, and give it budgets and tracing from day one.

{{< check q="An assistant answers questions from one collection of HR policy documents. Is an agent a good first design?" options="Yes, because agents are more accurate|Probably not: start single-shot, measure, and add a router or agent only if questions need several sources or steps|Yes, as long as it can also search the web" answer="2" explain="With one document source, an agent mainly adds latency, cost and new failure modes. A single-shot pipeline is easier to test, and the measurements tell you whether anything more is needed." >}}

---

## Layer 3: Quality Control

### Reranking and thresholds

A **cross-encoder reranker** reads the question and each candidate passage *together* and scores how well the passage answers the question. It is slower than vector search but usually more precise, so the common pattern is: retrieve a wider set (say, the top 20), rerank, keep the best few, and refuse to answer when no passage passes a minimum score.

**A small experiment: are reranker scores stable enough for a fixed threshold?** The question "Who won the 2022 football World Cup?" was scored against the same unrelated passage ("Keep this medicine out of the reach of children."), alone and together with other medical passages, using FlashRank 0.2.10:

| Batch size | `ms-marco-MultiBERT-L-12` | `ms-marco-MiniLM-L-12-v2` |
| ---: | ---: | ---: |
| 1 passage | 0.2126 | 0.0000 |
| 10 passages | **0.6877** | 0.0000 |
| 20 passages | 0.0028 | 0.0000 |

With the first model, the score of the *same* irrelevant passage changed from 0.003 to 0.69 depending on what else was in the batch, so a fixed threshold such as 0.05 would sometimes accept it. The second model scored it near zero every time. This is one small test with one library version, so it says nothing about how common the problem is. It does show why it is worth checking, **before relying on a score threshold, that the scores are stable and calibrated** on your own questions, including off-topic ones.

### Corrective and self-correcting retrieval

"Self-correcting RAG" is a family of ideas rather than one method. Two well-known research approaches:

* **Corrective RAG (CRAG)** (Yan et al., 2024) adds a lightweight *retrieval evaluator* that estimates whether the retrieved documents are relevant, and triggers one of three actions:
  * **Correct:** refine the retrieved documents, keeping only the useful parts;
  * **Incorrect:** discard them and fall back to web search;
  * **Ambiguous:** combine both sources.
* **Self-RAG** (Asai et al., 2023) trains the language model itself to decide when to retrieve, and to critique its own output with special *reflection* tokens (is this passage relevant, is this claim supported, is this answer useful).

In practical systems, the same family includes simpler pieces: query rewriting and a second search before giving up, reranker thresholds with refusal, and an **LLM judge** that scores whether the final answer is supported by the retrieved context.

These checks are useful, but they have limits worth stating clearly:

* **Relevant is not the same as sufficient.** A passage can be on-topic and still not contain the answer, or contradict another source.
* **An LLM judge is not independent fact-checking.** A model grading another model's answer can approve unsupported claims; judges also show biases such as preferring longer answers (Zheng et al., 2023).
* **Thresholds trade errors against refusals.** Stricter settings reduce wrong answers but increase refusals of good questions. Both rates need measuring.

In high-stakes domains such as healthcare, these layers add useful checkpoints, but they do not make a system safe on their own. That also needs domain-specific validation, clear limits on how the system may be used, and human review.

> **Quality control in one line:** verify before answering, calibrate thresholds on real questions, and treat automated checks as triage, not proof.

{{< check q="Your reranker gives an obviously off-topic passage a score of 0.69. What is the most useful next step?" options="Raise the threshold above 0.69 and move on|Check whether the scores are stable across batch sizes and calibrated on your own questions, including off-topic ones|Nothing: a cross-encoder is more accurate than vector search, so trust it" answer="2" explain="A threshold is only as reliable as the scores behind it. In the experiment above, the same passage scored anywhere from 0.003 to 0.69 depending on the batch, so first check the scores, then choose a threshold." >}}

{{< check q="An LLM judge marks an answer as faithful. Does that prove the answer is correct?" options="Yes, that is what faithfulness means|No: it only checks the answer against the retrieved text, which can be incomplete or wrong, and judges have known biases" answer="2" explain="Faithfulness means the answer is supported by the retrieved context. If retrieval missed a key document, a faithful answer can still be wrong. Use judges as triage, together with good retrieval and human review." >}}

---

## Putting It Together: A Reference Architecture

Here is one way the layers can fit together. It is a starting point to adapt, not the only correct design:

```text
[question + user identity]
        |
[access control: which documents may this user see?]   -> applied as a filter to every retrieval
        |
[router: what does this question need?]
        |-- documents only --> [hybrid retrieval: vector + BM25, merged with RRF]
        |-- structured data -> [Text-to-SQL, read-only]
        |-- several sources -> [agent with tools, step and time budgets]
        v
[rerank, keep the best passages above a calibrated threshold]  -> nothing left: refuse
        |
[generate an answer with citations]
        |
[verify: is the answer supported by the cited passages?]  -> not supported: refuse or retry
        |
[log tokens, cost and latency; cache only grounded answers]
```

A few details matter more than they look:

* **The router should be evaluated like any model.** Question length and keywords are cheap signals, but a short question ("Why did revenue fall?") can be hard, and a long one can be easy. Measure the router's decisions on labeled questions.
* **Caching needs scoping.** A cached answer must only be reused for the same question, the same allowed documents and the same document versions. In multi-user or patient-specific systems, a cache that ignores permissions can leak data.
* **Security belongs in the architecture**, not as an afterthought: authentication, permission-aware retrieval, audit logs, careful handling of personal data, and treating retrieved text as untrusted input.

---

## How to Evaluate a RAG System

Without measurement, every choice in this post is an educated guess. A useful evaluation looks at each layer separately, so failures can be traced to their source:

| Layer | What to measure | Common metrics |
| :--- | :--- | :--- |
| Retrieval | Is the right passage found? | Recall@k, MRR, nDCG |
| Reranking | Is it ranked near the top? | MRR, nDCG, before vs. after reranking |
| Generation | Is the answer correct and supported? | Answer correctness, faithfulness |
| Citations | Do the citations support the claims? | Citation precision and recall |
| Refusals | Does it refuse the right questions? | Wrong-answer rate vs. refusal rate |
| Operations | Is it affordable and fast enough? | Latency (p50/p95), tokens and cost per question |

A practical process:

1. Collect 50-200 **real questions** (or realistic ones written by people who know the domain), each labeled with the passage that answers it. Include some questions the documents *cannot* answer.
2. Measure a simple baseline first.
3. Change **one thing at a time** (add BM25, add a reranker, change chunk size) and compare against the baseline.
4. Keep a change only if it improves the metric you care about without unacceptable cost or latency.

This also answers the question "should I use hybrid search?" more reliably than a general rule can: try it on your labeled questions and see.

---

## How to Choose: A Decision Framework

The picker below asks about your questions, your identifiers and the cost of a wrong answer. Treat its suggestion as a starting point to test, not a final answer.

{{< rag-picker >}}

The same guidance as a table:

| If most questions... | A reasonable starting point | Then check |
| :--- | :--- | :--- |
| ask about meaning, in varied wording | Vector retrieval | Recall on paraphrased questions |
| contain codes, IDs, names or clause numbers | BM25 with identifier normalization, often plus vector search | Recall on exact-ID questions |
| mix both | Hybrid retrieval with RRF | Whether hybrid beats each method alone on your set |
| need numbers from tables | SQL with a read-only connection | Query correctness, permission checks |
| follow relationships between entities | A curated graph or database first; LLM-built graphs if needed | Entity resolution and extraction errors |
| need several sources or dependent steps | A router or agent with budgets | Tool-choice accuracy, latency, cost |
| would cause harm if answered wrongly | Any of the above, plus reranking thresholds, verification and human review | Wrong-answer rate vs. refusal rate |

### Other approaches worth knowing

Several techniques fit into the same three layers and deserve their own articles:

* **Retrieval:** *multi-query retrieval* (search with several rephrasings of the question and merge the results); *contextual retrieval* (Anthropic, 2024: add a short description of the document to each chunk before indexing); *late chunking* (embed the whole document first, then split the token vectors into chunks); *hierarchical (parent-child) retrieval*; *multimodal retrieval* over images and tables.
* **Orchestration:** *multi-hop retrieval* (each search uses the result of the previous one); *temporal retrieval* (prefer recent information when facts change over time).

---

## Conclusion

RAG is a design space, not a single pipeline. Separating it into three layers (how you retrieve, how you coordinate, and how you verify) makes the trade-offs easier to see, and makes it clear that most "types of RAG" are combined rather than chosen between.

The small experiments in this post also show that common one-line advice needs care: different codes can look more similar to an embedding model than true synonyms do; BM25 finds exact terms only if tokenization and normalization are right; and a reranker threshold is only as reliable as the reranker's scores.

The most dependable path is usually the least exciting one: start simple, build a labeled set of real questions, measure each layer, and add complexity where the measurements show it is needed.

---

### Reproduce the experiments

<details>
<summary>Python code for the three checks (click to expand)</summary>

Tested with Python 3.12, sentence-transformers 6.2.0, rank_bm25 0.2.2 and FlashRank 0.2.10. Models download on the first run.

```python
import re

import numpy as np
from flashrank import Ranker, RerankRequest
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 1. Embedding similarity: different codes vs. true synonyms
pairs = [
    ("E11.9", "E10.9"),
    ("Type 2 diabetes mellitus without complications (E11.9)",
     "Type 1 diabetes mellitus without complications (E10.9)"),
    ("heart attack", "myocardial infarction"),
    ("metformin", "metoprolol"),
    ("heart attack", "football match"),
]
for name in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    model = SentenceTransformer(name)
    for a, b in pairs:
        va, vb = model.encode([a, b], normalize_embeddings=True)
        print(f"{name}: {float(np.dot(va, vb)):.3f}  {a} | {b}")

# 2. BM25 depends on tokenization
docs = ["Patient diagnosed with E11.9, started on metformin 500 mg.",
        "Type 1 diabetes (E10.9) managed with insulin.",
        "Hypertension follow-up in 4 weeks."]
naive = BM25Okapi([d.lower().split() for d in docs])
print("whitespace, 'E11.9':", naive.get_scores(["e11.9"]))

def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)*", text.lower())  # keep codes like e11.9 together

better = BM25Okapi([tokenize(d) for d in docs])
print("code-aware, 'E11.9':", better.get_scores(tokenize("E11.9")))
print("code-aware, 'E119': ", better.get_scores(tokenize("E119")))

# 3. Is a reranker score stable across batch sizes?
query = "Who won the 2022 football World Cup?"
target = "Keep this medicine out of the reach of children."
fillers = [
    "Amlodipine: initial dose 5 mg once daily for hypertension.",
    "Maximum dose of amlodipine is 10 mg once daily.",
    "Dose adjustment may be needed in hepatic impairment.",
    "Follow-up visits are recommended every 4 weeks.",
    "Do not drive if you feel dizzy after taking this medicine.",
    "Store below 25 degrees Celsius.",
    "Common side effects include ankle swelling and flushing.",
    "Check blood pressure before each dose increase.",
    "Tell your doctor about all other medicines you take.",
    "Grapefruit juice may increase blood levels of some medicines.",
    "Report chest pain or fainting immediately.",
    "Use with caution in patients with heart failure.",
    "Children under 6 years: safety not established.",
    "Take the tablet at the same time each day.",
    "Missed dose: take it as soon as you remember.",
    "Overdose may cause very low blood pressure.",
    "Do not stop taking it without medical advice.",
    "Pregnancy: discuss risks and benefits with your doctor.",
    "Elderly patients may start with a lower dose.",
]
for name in ["ms-marco-MultiBERT-L-12", "ms-marco-MiniLM-L-12-v2"]:
    ranker = Ranker(model_name=name)
    for n in (0, 9, 19):
        passages = [{"id": 0, "text": target}] + [{"id": i + 1, "text": f} for i, f in enumerate(fillers[:n])]
        scores = {r["id"]: r["score"] for r in ranker.rerank(RerankRequest(query=query, passages=passages))}
        print(f"{name}: batch of {n + 1} -> {float(scores[0]):.4f}")
```


</details>

### Further reading

* Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). NeurIPS 2020.
* Robertson, S., & Zaragoza, H. (2009). [The Probabilistic Relevance Framework: BM25 and Beyond](https://doi.org/10.1561/1500000019). Foundations and Trends in Information Retrieval.
* Cormack, G., Clarke, C., & Buettcher, S. (2009). [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods](https://doi.org/10.1145/1571941.1572114). SIGIR 2009.
* Edge, D., et al. (2024). [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130). Microsoft Research.
* Jeong, S., et al. (2024). [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403). NAACL 2024.
* Yan, S., et al. (2024). [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884). arXiv preprint.
* Asai, A., et al. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). arXiv preprint (ICLR 2024).
* Greshake, K., et al. (2023). [Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173). arXiv preprint.
* Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks.
* Gao, Y., et al. (2023). [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997). arXiv preprint.
* Anthropic (2024). [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval).

---

## Let's Discuss

If you have measured any of these trade-offs on your own data (hybrid vs. vector retrieval, reranker thresholds, or agents vs. fixed workflows), I would like to learn from what you found, including results that disagree with this post. Corrections are welcome too.

<div class="author-card">
  <div class="author-card-title">Connect & Collaborate</div>
  <p>I am learning and writing about RAG systems and retrieval quality. If you are building one and want to compare notes, I would be glad to hear from you.</p>
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
