---
title: "Why Naive RAG Fails in Healthcare: 4 Failure Modes and How to Fix Them"
date: 2026-04-28T14:30:00+05:30
lastmod: 2026-11-12T12:00:00+05:30
description: "Four ways naive RAG fails on clinical documents, and the fixes: OCR fallback, parent-child chunking, cross-encoder reranking and LLM-as-judge evaluation."
summary: "A practical guide to why simple Retrieval-Augmented Generation (RAG) breaks on real clinical documents, and the design patterns that fix it: OCR fallback, parent-child chunking, cross-encoder reranking and automated evaluation."
keywords: ["Clinical RAG", "Healthcare AI", "Naive RAG Failures", "Parent-Child Chunking", "Cross-Encoder Reranking", "LLM-as-Judge", "OCR", "LLMOps"]
tags: ["Clinical RAG", "Healthcare", "NLP", "Vector Databases", "Inference", "LLMOps"]
TocOpen: true
draft: false
---

> *Disclaimer: This article discusses engineering considerations for clinical AI systems. It is for educational purposes only and is not medical advice, clinical guidance, or regulatory compliance counsel.*

Picture this (a hypothetical example). A patient with chronic kidney disease arrives with acute pain. The hospital's RAG assistant reads their records, including an old scanned fax. One low-contrast page of that fax lists a Stage 4 kidney impairment, but the PDF parser returns empty text for it, and nobody notices. Asked what pain relief is suitable, the assistant confidently suggests a high-dose NSAID such as ibuprofen. NSAIDs can seriously harm failing kidneys.

The language model did nothing wrong. It answered correctly *from what it was given*. The pipeline failed long before the model was called.

That is the theme of this post: **most RAG failures in healthcare happen before the model runs.** Here are the four most common ones, and how to fix each.

> **TL;DR: the four failures and their fixes**
>
> | # | Failure | Fix |
> | :--- | :--- | :--- |
> | 1 | Scanned pages silently become empty text | Extract text first, fall back to OCR for blank or garbled pages, and flag every OCR'd chunk |
> | 2 | Fixed-size chunks split clinical facts and dilute rare terms like ICD codes | Parent-child chunking: search small chunks, send their larger parent passages to the model |
> | 3 | Relevant evidence gets buried in the middle of the prompt | Cross-encoder reranking, best passage first, and refusing when nothing is relevant enough |
> | 4 | "Looks fine to me" evaluation hides whether retrieval or generation failed | An LLM judge scoring faithfulness, answer relevance and context precision, backed by human review |

---

## Why Healthcare Data Is Hard

Clinical information is spread across hospitals, clinics, labs and insurers, and none of them share a format. One patient's diagnosis may sit in a typed PDF, their other conditions in a scanned fax, and their lab values in an Excel export. This is normal, not an edge case.

The consequence: **model quality cannot make up for a broken index.** If a page was silently dropped during ingestion, the model has no way to know it ever existed, and it will answer confidently anyway. So every fix below starts from one question: *what can go missing, and how would we notice?*

---

## Failure 1: Scanned Pages Silently Disappear

**Analogy:** imagine a librarian who indexes every book by reading its text, but treats every photocopied page as blank. The catalogue looks complete. Whole chapters are missing from it.

### What goes wrong
Common PDF libraries (such as `pdfplumber`, `pypdf`, or LangChain's default `PyPDFLoader`) read the text layer of a PDF. A scanned page has no text layer; it is just an image. So the library returns an empty string, with no error. The page is then either skipped or stored as an empty chunk, and its content is gone.

Clinical records are full of such pages: scanned referral letters, photographed pathology reports, faxed discharge summaries. A clinician can ask about a patient's surgical history and get "no relevant information found", simply because that history lived on a scanned page.

### The fix: text extraction first, OCR as a fallback
* **Stage 1:** extract text from every page with a fast library such as PyMuPDF.
* **Stage 2:** when a page comes back blank or garbled, render it as an image and run OCR (for example, EasyOCR). If the result is still empty, retry at a higher resolution.

```python
# Two-stage extraction (simplified)
page_text = extract_with_pymupdf(page)
ocr_used = False

if not page_text.strip() or looks_garbled(page_text):
    ocr_used = True
    page_text = run_ocr(rasterize_page(page, zoom=2))      # about 144 DPI
    if not page_text.strip():
        page_text = run_ocr(rasterize_page(page, zoom=3))  # about 216 DPI, retry

if not page_text.strip():
    log.warning("Page %d unreadable after OCR", page.number)  # flag it, never drop it silently
```

Store an `ocr_used` flag on every chunk that came from OCR. When an answer looks wrong later, you can see at once whether its evidence came from OCR text.

**Trade-off:** OCR is slow (seconds per page on a CPU, versus milliseconds for text extraction). Run ingestion in the background, through a job queue such as Celery or RQ at scale, so users never wait for it.

> **Remember:** an empty string is not an error, so nothing warns you. Check every page, fall back to OCR, and flag what you could not read.

---

## Failure 2: Fixed-Size Chunks Blur the Details

**Analogy:** if you summarize a whole textbook page in one sentence, the rare drug name in paragraph four does not survive. Large chunks do the same thing to embeddings.

### What goes wrong
Many embedding models (MiniLM, for example) turn a piece of text into one vector by averaging the vectors of its tokens. When documents are cut into fixed blocks (say, 512 tokens), two things go wrong.

First, a cut can fall in the middle of a fact. In *"Type 2 Diabetes Mellitus (ICD-10: E11.9), hypertensive urgency (ICD-10: I16.0), and stage 3 chronic kidney disease (ICD-10: N18.3)"*, a boundary can separate a diagnosis from its code.

Second, rare but important terms (ICD codes, unusual drug names) are a tiny share of a large chunk's tokens. The averaged vector is dominated by common phrasing such as "patient presents with". This **semantic dilution** means a search for a specific condition can miss the chunk that mentions it.

### The fix: parent-child chunking
Separate *what you search* from *what you send to the model*:

* **Child chunks (about 300-500 characters):** small enough to hold one clinical statement. Only these are embedded and searched.
* **Parent chunks (about 1,500-2,000 characters):** stored separately. Each child records the id of its parent.
* **At query time:** find the best children, then swap each one for its parent, so the model gets enough context to answer well.

```python
# Parent-child retrieval (simplified)
children = vector_db.similarity_search(query, k=10)
context = [parent_store[child.metadata["parent_id"]] for child in children]
```

Small chunks give precise matches; large parents give context. You get both, at the cost of one extra lookup per query.

### Going further
* **Hybrid search:** add a keyword index (BM25) next to the vector index and merge the two rankings with Reciprocal Rank Fusion (RRF). Keyword search never dilutes an exact ICD code.
* **Structured data:** values that change over time (for example, a year of HbA1c results) belong in a database query, not in vector search.
* **Recency:** for many questions, yesterday's note matters more than a five-year-old one, but not for all (allergies stay relevant). A **temporal decay** factor, *score × e^(−λt)* with *t* the document's age in days, lets you tune this per data type.
* **Other chunking strategies** worth knowing: ColBERT (token-level matching), late chunking, and RAPTOR (a tree of summaries).

> **Remember:** search with small chunks for precision, answer with large parents for context.

---

## Failure 3: The Right Evidence, in the Wrong Place

**Analogy:** hand a busy doctor a 20-page file where the critical allergy note is on page 11. They will read page 1 carefully, skim the end, and probably miss page 11.

### What goes wrong
A naive pipeline pastes retrieved chunks into the prompt in plain similarity order. A key detail, such as a medication allergy, can land in the middle of a long context, where the model may overlook it.

Two things cause this. First, **models do not read long prompts evenly**: Liu et al. ("Lost in the Middle", 2023) found that models use information at the start and end of a long context better than information in the middle. Second, **similarity is not relevance**: a general paragraph about pain relief can score higher than the one sentence saying "NSAIDs are contraindicated for this patient".

### The fix: rerank with a cross-encoder
Add a **cross-encoder reranker** (for example, FlashRank or Cohere Rerank) right after vector search:

```text
Vector search (top 10 small chunks) ➔ Cross-encoder rerank ➔ keep top 3 ➔ swap in parents ➔ LLM
```

A cross-encoder reads the question and each passage *together*, so it judges relevance far better than comparing two separate vectors. Then:

* put the most relevant passage **first**, where the model is most likely to use it;
* drop passages below a minimum relevance score;
* if nothing is left, **refuse** ("I could not find this in the documents") instead of letting the model guess;
* cap the context with a token budget (for example, 6,000 tokens), so prompts stay focused and costs predictable.

**Trade-off:** a cross-encoder runs a model on every question-passage pair, so it is slower than vector search. Reranking only the top 10 candidates keeps it manageable; measure the cost on your own hardware.

> **Remember:** retrieval finds candidates; reranking decides what the model reads first, and whether it should answer at all.

---

## Failure 4: "It Looks Fine to Me" Is Not Evaluation

**Analogy:** grading exam papers by glancing at the handwriting. You will form an opinion, but you cannot say *which* answers were wrong, or why.

### What goes wrong
Prototypes are usually tested by eye: ask a few questions, read the answers, nod. This does not scale, varies between reviewers, and misses slow quality drift after new documents arrive. Worse, the final answer rarely shows *where* things broke: a **retrieval failure** (good reasoning over the wrong chunks) and a **generation failure** (hallucination over the right chunks) look the same.

### The fix: an LLM judge with three scores
Score every answer with a separate "judge" model at `temperature=0` (low variance, although hosted models are not perfectly deterministic). It sees the question, the retrieved context and the answer, and returns three scores from 0 to 1:

* **Faithfulness:** is every claim supported by the context? A low score means hallucination.
* **Answer relevance:** does the answer address the question?
* **Context precision:** was the retrieved context useful? A low score points to a retrieval problem.

```python
# Turning three scores into a grade (illustrative thresholds)
composite = (faithfulness + answer_relevance + context_precision) / 3
grade = "A" if composite >= 0.85 else "B" if composite >= 0.70 else "C" if composite >= 0.50 else "F"
```

Log low-scoring answers with their chunk ids, rerank scores and OCR flags, so you can trace each failure back to the step that caused it.

### Judges make mistakes too
Zheng et al. (2023) found that LLM judges show **position bias**, **verbosity bias** and **self-enhancement bias** (favoring answers similar to their own). So treat the judge as triage, not as the final word: it decides which answers a human should review first.

A judge also adds a model call to every question. You can run it in the background, but then it cannot stop a bad answer. If low scores should **block** an answer (show a safe "please check the source documents" message and send the case to a clinical review queue instead), the judge must run first, and every user pays the extra latency. Choose per use case.

Finally, every clinical statement in an answer should carry a **citation** (file, page, original text), so a clinician can verify it in seconds.

> **Remember:** score retrieval and generation separately, so you know which one failed, and keep a human in the loop.

---

## Where Does Patient Data Go?

All four fixes raise the same question. Cloud OCR (such as AWS Textract or Google Document AI), cloud rerankers (such as Cohere) and hosted language models are all strong and easy to run, but each one sends clinical text outside your network.

That is only acceptable with the right legal agreements (for example, a Business Associate Agreement under HIPAA in the US) and controls. Otherwise, keep those steps on your own servers, or **de-identify** text before it leaves. Tools like Microsoft Presidio detect names, dates and locations; simple regex rules catch structured identifiers (phone numbers, record numbers) but not names reliably. Whether a tool counts as Software as a Medical Device under FDA rules also depends on its intended use.

---

## Putting It Together: A Reference Query Flow

One reasonable order of steps for answering a question. Step through it yourself: pick a scenario and press **Next** (or **Play**). The documents, scores and answers are illustrative examples.

{{< rag-flow >}}

The same flow as a plain diagram. The numbers are typical starting values to tune, not fixed rules:

```text
[Question + chat history]
        ↓
[0. Rewrite a follow-up into a standalone question (only when there is chat history)]
        ↓
[1. Exact-match cache: hash of the normalized question + document filter] ➔ hit: return the stored answer
        ↓ miss
[2. Vector search: top 10 small child chunks]
        ↓
[3. Cross-encoder rerank: keep the top 3 above a minimum score] ➔ nothing left: refuse
        ↓
[4. Swap each child for its parent chunk, best first; trim to a token budget]
        ↓
[5. Route: a smaller model for short, simple questions; a larger one for the rest]
        ↓
[6. Answer with citations]
        ↓
[7. Judge the answer]
        ↓
[8. Record tokens and cost; cache only grounded answers, never refusals]
```

Each step comes from a fix above: steps 2 and 4 are parent-child chunking (Failure 2), step 3 is reranking and refusal (Failure 3), step 7 is the judge (Failure 4), and OCR (Failure 1) already happened at ingestion time, before any question is asked.

Three choices here are easy to get wrong:

* **Rewrite before the cache.** "What about its dose?" means different things in different conversations. Rewriting it into a full question first ("What is the dose of amlodipine?") makes the cache key correct.
* **Exact-match cache, not a semantic cache.** Terms that look similar to an embedding model can mean opposite things (hypokalemia and hyperkalemia). An exact hash never returns the answer to a different question.
* **Route by difficulty, then verify.** A smaller model can cost a fraction of a larger one. Simple rules (question length, keywords such as "contraindication", context size) are a cheap start, but check the routing against real questions before trusting the savings.

---

## Every Layer Has a Cost

Each fix adds something to build, run and monitor: an OCR worker, a task queue, a reranker, a cache, a de-identification step. Each is also one more thing that can fail, so each needs a fallback. For a general-purpose assistant, much of this is over-engineering; in clinical settings, much of it is justified by patient-safety risk. Either way, **start with the layers that fix failures you have actually measured.**

---

## Before You Ship: A Clinical RAG Checklist

- [ ] **(Failure 1)** Every page is either extracted, OCR'd, or logged as unreadable; nothing is dropped silently.
- [ ] **(Failure 1)** Chunks carry metadata: source file, page or section, and whether OCR was used.
- [ ] **(Failure 2)** Small chunks are searched; larger parent passages are sent to the model.
- [ ] **(Failure 2)** Exact terms (ICD codes, drug names) can be found, through hybrid search if needed.
- [ ] **(Failure 3)** Results are reranked, the best evidence comes first, and the context has a token budget.
- [ ] **(Failure 3)** The system refuses when no evidence passes the relevance threshold.
- [ ] **(Failure 4)** Every answer carries citations a clinician can check.
- [ ] **(Failure 4)** Answers are scored automatically, and low scores reach a human reviewer.
- [ ] **(Patient data)** You know where patient data travels, and every external API is covered by an agreement or receives de-identified text.
- [ ] **(Operations)** Token usage and cost are tracked per question.
- [ ] **(Operations)** A test set of real questions with expected sources is re-run after every change.

---

## Conclusion

Remember the patient from the start? Their kidney diagnosis sat on a scanned page that came back empty. The first checklist item (never drop a page silently) would have flagged that page, and citations would have shown the clinician exactly which records the answer relied on, making the missing kidney history much easier to spot.

The steps of a RAG pipeline form a chain, and a failure early in the chain travels all the way to the end. A silently dropped page produces no chunk, no context and no evidence. Worse, the judge may still score the final answer as faithful, because it only sees what was retrieved, not what was lost. **Evaluation at the end is only as good as data integrity at the start.**

Clinical RAG is as much a systems engineering problem as a model problem. Get the pipeline right, and even a modest model gives safer answers than a brilliant model reading a broken index.

---

### Further Reading

* Liu, N. F., et al. *Lost in the Middle: How Language Models Use Long Contexts*. Transactions of the ACL (2024); preprint 2023.
* Nogueira, R., & Cho, K. (2019). *Passage Re-ranking with BERT*. arXiv preprint.
* Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*. NeurIPS 2023 Datasets and Benchmarks.
* Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.

---

## 💬 Community Discussion

Join the ongoing discussion and share your thoughts on this architecture on Reddit:

<blockquote class="reddit-embed-bq" style="height:316px" data-embed-height="316">
<a href="https://www.reddit.com/r/Rag/comments/1wnzmap/why_naive_rag_fails_in_production_healthcare_a/">Why Naive RAG Fails in Production Healthcare: A Deep Dive into Parent-Child Chunking, Cross-Encoders &amp; Clinical Evaluation</a><br> by
<a href="https://www.reddit.com/user/JhaDevesh/">u/JhaDevesh</a> in
<a href="https://www.reddit.com/r/Rag/">Rag</a>
</blockquote><script async="" src="https://embed.reddit.com/widgets.js" charset="UTF-8"></script>

---

<div class="author-card">
  <h3>Connect & Collaborate</h3>
  <p>Always open to discussing RAG architectures, edge cases in clinical NLP, or exchanging notes on production AI systems. Interested in collaborating or chatting about engineering challenges? Let's connect!</p>
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
