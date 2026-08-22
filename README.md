---
title: APIcalypse Voice RAG
emoji: 🎙️
colorFrom: orange
colorTo: red
sdk: docker
app_port: 7860
short_description: Grounded, sub-25ms multilingual voice RAG over MSMARCO-XI
---

<div align="center">

# 🎙️ APIcalypse Voice RAG

### Speak. Trust.
**A grounded multilingual voice interface for retrieval-based question answering over a curated knowledge base.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-HTTPS_Active-brightgreen?style=for-the-badge&logo=googlechrome)](https://planned-controlled-alpha-expects.trycloudflare.com)
[![Custom Domain](https://img.shields.io/badge/Custom_Domain-apicalypsevoicerag.me-blue?style=for-the-badge)](https://apicalypsevoicerag.me)
[![Swagger Docs](https://img.shields.io/badge/Swagger_UI-FastAPI_Docs-orange?style=for-the-badge&logo=fastapi)](https://planned-controlled-alpha-expects.trycloudflare.com/docs)
[![Tests](https://img.shields.io/badge/Tests-104%2F104%20PASS-3dff8a?style=for-the-badge&logo=pytest)](hhg-task2/docs/TESTING.md)
[![RAG P50](https://img.shields.io/badge/RAG_P50-22_ms-0e241b?style=for-the-badge&logo=speedtest)](hhg-task2/docs/LATENCY.md)
[![RAG P100](https://img.shields.io/badge/RAG_P100-27_ms_%3C_100-3dff8a?style=for-the-badge)](hhg-task2/docs/LATENCY.md)
[![Languages](https://img.shields.io/badge/Languages-EN%20%7C%20HI%20%7C%20GU%20%7C%20Indic-ffb020?style=for-the-badge)](hhg-task2/src/utils/language.py)
[![Corpus](https://img.shields.io/badge/MSMARCO--XI-15%2C680_Chunks-6f42c1?style=for-the-badge)](hhg-task2/data/)
[![Track](https://img.shields.io/badge/Track-%23RAGInGoa-ff5500?style=for-the-badge)](#ragingoa)

</div>

---

## 🌐 Live Production Deployment

* 🎙️ **Live Web Application (HTTPS):** [https://planned-controlled-alpha-expects.trycloudflare.com](https://planned-controlled-alpha-expects.trycloudflare.com)
* 🌐 **Custom Domain (SSL):** [https://apicalypsevoicerag.me](https://apicalypsevoicerag.me)
* 📖 **Interactive Swagger API Docs:** [https://planned-controlled-alpha-expects.trycloudflare.com/docs](https://planned-controlled-alpha-expects.trycloudflare.com/docs)
* 📊 **Live Index Stats:** [https://planned-controlled-alpha-expects.trycloudflare.com/api/stats](https://planned-controlled-alpha-expects.trycloudflare.com/api/stats)

---

## 🚀 Quick Navigation

The complete Task #2 implementation, web application, test suite, and forensic documentation are located in [`hhg-task2/`](hhg-task2/).

- **Live Application:** [https://planned-controlled-alpha-expects.trycloudflare.com](https://planned-controlled-alpha-expects.trycloudflare.com)
- **Executive Overview & Demo Guide:** [`hhg-task2/README.md`](hhg-task2/README.md)
- **Automated Test Suite:** Run `pytest tests/ -v` (**104/104 PASS**)

---

## ⚡ Measured RAG Latency Summary

| Stage | P50 (Median) | P70 | P95 | P100 (Max) | Target Budget |
| :--- | :---:| :---:| :---:| :---:| :--- |
| **Query Embedding** | **10.59 ms** | 11.40 ms | 13.67 ms | 16.50 ms | `< 30 ms` |
| **Vector Retrieval + Reranking** | **10.73 ms** | 11.33 ms | 12.57 ms | 13.80 ms | `< 30 ms` |
| **6-Layer Guardrails** | **0.34 ms** | 0.39 ms | 0.43 ms | 0.54 ms | `< 5 ms` |
| **Answer Generation** | **0.00 ms** | 0.00 ms | 0.00 ms | 0.00 ms | `< 10 ms` |
| **TOTAL RAG PIPELINE** | **`22.76 ms`** | **`23.84 ms`** | **`26.67 ms`** | **`31.33 ms`** | **`< 100 ms`** |

---

## 📚 Forensic Documentation Tree

- 🏛️ **[hhg-task2/docs/ARCHITECTURE.md](hhg-task2/docs/ARCHITECTURE.md)** — Architectural design & component diagrams.
- ⏱️ **[hhg-task2/docs/LATENCY.md](hhg-task2/docs/LATENCY.md)** — Stage timing methodology & CPU optimizations.
- 🧪 **[hhg-task2/docs/TESTING.md](hhg-task2/docs/TESTING.md)** — 99-test suite breakdown & validation procedures.
- 🛡️ **[hhg-task2/docs/GUARDRAILS.md](hhg-task2/docs/GUARDRAILS.md)** — 6 guardrail layers & the NOAA fix.
- ⚖️ **[hhg-task2/docs/DECISIONS.md](hhg-task2/docs/DECISIONS.md)** — Engineering tradeoffs & design rationale.

---

*Built with pride by **APIcalypse** for Hacker House Goa 2026 (#RAGInGoa).*
