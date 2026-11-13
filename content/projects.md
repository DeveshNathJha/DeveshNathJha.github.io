---
title: "Projects"
description: "Projects by Devesh Nath Jha: a healthcare RAG system and an edge-AI fall-detection system."
ShowReadingTime: false
ShowToc: false
ShowBreadCrumbs: false
ShowPostNavLinks: false
# TODO: add live demo URLs to the "Links" lines once the projects are deployed.
---

## Healthcare RAG: question answering over medical documents

A web app that answers questions about uploaded medical documents (PDF, CSV, Excel) and cites the
file and page for every answer. If the documents do not contain the answer, it says so instead of
guessing.

**Links:** [Source code on GitHub](https://github.com/DeveshNathJha/Healthcare-Rag) ·
[Blog post: Why Naive RAG Fails in Healthcare](/posts/why-naive-rag-fails-healthcare/)

**How it works**

* **Ingestion:** PyMuPDF reads digital PDFs; scanned or garbled pages fall back to EasyOCR.
  Spreadsheets are split into labelled row sections, so citations can point to a row range.
* **Retrieval:** small-to-big chunking (small chunks are searched, their larger parent passages are
  sent to the model), MiniLM embeddings, FAISS search and a FlashRank cross-encoder reranker.
  Answers are refused when the best match is too weak.
* **Answering:** Groq-hosted LLMs, with a simple router that sends short, simple questions to a
  smaller model and the rest to a larger one. Follow-up questions are rewritten into standalone
  questions first.
* **Quality:** an LLM judge scores each answer for faithfulness, answer relevance and context precision.
* **Privacy:** optional masking of structured identifiers (phone numbers, ID numbers) before text
  is sent to the LLM.
* **Engineering:** FastAPI backend with API keys, rate limits, upload validation and background
  processing; a simple web UI; Docker image; GitHub Actions CI with 38 automated tests.

**What I learned:** most answer-quality problems came from ingestion and retrieval, not from the
language model.

**Status:** a portfolio project, built and tested with public sample documents. It is not used with
real patient data.

**Stack:** Python, FastAPI, LangChain, FAISS, FlashRank, sentence-transformers, EasyOCR, PyMuPDF,
Groq, Docker, GitHub Actions.

---

## Passenger Safety & Fall Detection System (Edge AI)

A system that detects human falls from smartphone or wearable motion sensors (accelerometer and
gyroscope) and can send an SOS alert.

**Links:** [Source code on GitHub](https://github.com/DeveshNathJha/Passenger-Safety-Fall-Detection-System) ·
[Live demo](https://passenger-safety-ai.onrender.com/demo) (free hosting: the first load can take
up to a minute while the server wakes up)

**How it works**

* **Model:** a 1D-CNN with about 8.5k parameters, quantized to a 20.5 KB TFLite model for edge
  devices. It reaches an F1 score of 0.97 (98% recall) on people it never saw during training.
* **Data:** MobiFall and SisFall, 62 people in total, including 15 aged 60 to 75. Train, validation
  and test sets never share a person, so the score reflects new users.
* **Robustness:** rotation-invariant input features, so the F1 score stays the same (0.97) even
  when the phone is randomly rotated.
* **Tested like real use:** replaying whole test recordings, the system caught 431 of 435 falls.
* **Serving:** a FastAPI WebSocket server streams sensor data and runs the model per session. A
  post-fall stillness check and a cancellable countdown reduce false alarms before an SMS alert
  (Twilio) is sent. Dockerized, with GitHub Actions CI and 111 automated tests.
* **Vehicle crashes:** a second model is trained on physics-based synthetic crash data, because no
  public labelled smartphone crash dataset exists.

**What I learned:** my first version had data leakage between training and test people. Fixing the
evaluation mattered more than changing the model.

**Stack:** Python, TensorFlow/Keras, TFLite, NumPy, FastAPI, WebSockets, Twilio, Docker, GitHub Actions.
