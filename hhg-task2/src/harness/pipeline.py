"""
RAG Pipeline — LangGraph State Machine.

The core orchestration pipeline wiring together:
  transcribe → retrieve → guardrail_check → generate → grounding_check

Uses LangGraph for state machine transitions with conditional edges
for guardrail-triggered refusal paths.
"""

import time
import uuid
import structlog
import httpx

from src.harness.state import RAGState, PipelineStage
from src.harness.models import PipelineResult, LatencyBreakdown
from src.harness.retry import with_retry
from src.guardrails.models import RefusalReason
from src.guardrails.language_consistency import LanguageConsistencyGuardrail
from src.guardrails.answerability import AnswerabilityGuardrail
from src.utils.language import QueryObject, detect_language, compute_answerability, normalize_language
from src.config import settings
from src.generation.extractive import extractive_answer

logger = structlog.get_logger(__name__)


class RAGPipeline:
    """
    The main RAG pipeline orchestrator.

    Wires together STT, embeddings, retrieval, guardrails, and generation
    into a coherent pipeline with structured I/O and error handling.

    This is implemented as a simple async pipeline (no LangGraph dependency at this stage)
    for minimal overhead. Can be upgraded to LangGraph state machine in Chunk 5.
    """

    def __init__(
        self,
        stt_provider=None,
        embedding_service=None,
        vector_store=None,
        fast_store=None,
        llm_primary=None,
        llm_fallback=None,
        off_topic_guardrail=None,
        unsafe_guardrail=None,
        grounding_guardrail=None,
        coverage_guardrail=None,
        language_consistency_guardrail=None,
        answerability_guardrail=None,
    ):
        self.stt = stt_provider
        self.embeddings = embedding_service
        self.store = vector_store
        self.fast_store = fast_store
        self.llm_primary = llm_primary
        self.llm_fallback = llm_fallback
        self.off_topic = off_topic_guardrail
        self.unsafe = unsafe_guardrail
        self.grounding = grounding_guardrail
        self.coverage = coverage_guardrail
        self.language_consistency = language_consistency_guardrail or LanguageConsistencyGuardrail()
        self.answerability = answerability_guardrail or AnswerabilityGuardrail()

    async def process_voice(self, audio_bytes: bytes) -> PipelineResult:
        """
        Full voice-to-answer pipeline.

        Audio → STT → Query → Retrieval → Guardrails → Generation → Answer
        """
        trace_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        latency = LatencyBreakdown()

        try:
            # Stage 1: Transcribe
            stt_result = await with_retry(
                self.stt.transcribe,
                audio_bytes,
                max_retries=1,
                retryable_exceptions=(httpx.TransportError,),
            )
            latency.stt_ms = stt_result.duration_ms
            transcript = stt_result.transcript
            stt_lang = getattr(stt_result, "language", None)

            if not transcript.strip():
                return PipelineResult(
                    success=False,
                    refused=True,
                    refusal_reason=RefusalReason.SYSTEM_ERROR,
                    refusal_message="Could not understand the audio. Please try speaking again clearly.",
                    trace_id=trace_id,
                    latency=latency,
                )

            # Continue with text pipeline (measures pure RAG latency excluding STT)
            result = await self._process_text(
                transcript,
                latency,
                trace_id,
                input_language=normalize_language(stt_lang),
            )
            rag_total_ms = result.latency.total_ms
            e2e_ms = round((time.perf_counter() - start) * 1000, 2)
            result.transcript = transcript
            result.latency.stt_ms = latency.stt_ms
            result.latency.total_ms = rag_total_ms
            result.latency.e2e_ms = e2e_ms
            return result

        except Exception as e:
            total_ms = (time.perf_counter() - start) * 1000
            latency.total_ms = round(total_ms, 2)
            logger.error("pipeline_error", trace_id=trace_id, error=str(e))
            return PipelineResult(
                success=False,
                refused=True,
                refusal_reason=RefusalReason.SYSTEM_ERROR,
                refusal_message="An internal error occurred. Please try again.",
                trace_id=trace_id,
                latency=latency,
            )

    async def process_text(self, query: str, input_language: str | None = None) -> PipelineResult:
        """
        Text-only query pipeline for API testing and benchmarks.
        """
        trace_id = str(uuid.uuid4())[:8]
        latency = LatencyBreakdown()
        return await self._process_text(query, latency, trace_id, input_language=input_language)

    async def _process_text(
        self,
        query: str,
        latency: LatencyBreakdown,
        trace_id: str,
        input_language: str | None = None,
    ) -> PipelineResult:
        """Internal text processing pipeline."""
        pipeline_start = time.perf_counter()

        # 1. Detect and represent Language-Aware Query Object
        query_lang = normalize_language(input_language) or detect_language(query)
        query_obj = QueryObject(query=query, language=query_lang, raw_language=input_language)

        try:
            # Stage: Pre-generation guardrails (unsafe input)
            pre_guard_ms = 0.0
            if self.unsafe:
                t0 = time.perf_counter()
                unsafe_result = self.unsafe.check(query)
                pre_guard_ms += (time.perf_counter() - t0) * 1000
                if not unsafe_result.passed:
                    latency.guardrail_pre_ms = round(pre_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=unsafe_result.reason,
                        refusal_message=unsafe_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Stage: Query Embedding
            embed_start = time.perf_counter()
            query_embedding = self.embeddings.encode_query(query)
            latency.embedding_ms = round((time.perf_counter() - embed_start) * 1000, 2)

            if self.off_topic:
                t0 = time.perf_counter()
                off_topic_result = self.off_topic.check(query_embedding)
                pre_guard_ms += (time.perf_counter() - t0) * 1000
                if not off_topic_result.passed:
                    latency.guardrail_pre_ms = round(pre_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=off_topic_result.reason,
                        refusal_message=off_topic_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            latency.guardrail_pre_ms = round(pre_guard_ms, 2)

            # Stage: Retrieve (Always execute dense multilingual retrieval)
            retrieval_result = self.store.query(
                query_embedding,
                query_str=query,
                namespace=settings.retrieval_namespace,
                top_k=settings.retrieval_top_k,
            )
            retrieval_result.query = query
            latency.retrieval_ms = retrieval_result.duration_ms

            if not retrieval_result.chunks:
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResult(
                    query=query,
                    success=False,
                    refused=True,
                    refusal_reason=RefusalReason.UNGROUNDED,
                    refusal_message="No relevant information found in the knowledge base for your question.",
                    trace_id=trace_id,
                    latency=latency,
                )

            top_chunk = retrieval_result.chunks[0]
            top_score = float(top_chunk.score)
            doc_lang = top_chunk.metadata.get("language") or detect_language(top_chunk.text)
            fallback_used = top_chunk.metadata.get("fallback_used", False)

            # Calibrated retrieval floor: Indic languages have a calibrated baseline (0.40)
            INDIC_LANGS = {"gu", "ta", "te", "bn", "kn", "ml", "pa", "mr", "or"}
            grounding_threshold = (
                settings.grounding_threshold_gu if query_lang in INDIC_LANGS else settings.grounding_threshold
            )
            if top_score < grounding_threshold:
                logger.info("retrieval_below_semantic_threshold", top_score=round(top_score, 4), threshold=grounding_threshold, query=query[:100])
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResult(
                    query=query,
                    success=False,
                    refused=True,
                    refusal_reason=RefusalReason.UNGROUNDED,
                    refusal_message="No sufficiently grounded evidence was found in the knowledge base for this question.",
                    trace_id=trace_id,
                    latency=latency,
                )

            # Stage: Answerability Guardrail (Verifies specific question answerability)
            post_guard_ms = 0.0
            if self.answerability:
                t0 = time.perf_counter()
                ans_result = self.answerability.validate(query, top_chunk.text, query_language=query_lang)
                post_guard_ms += (time.perf_counter() - t0) * 1000
                if not ans_result.passed:
                    ans_score = top_chunk.metadata.get("answerability_score", 0.0)
                    logger.info(
                        "answerability_guard_refusal",
                        trace_id=trace_id,
                        query=query[:100],
                        query_language=query_lang,
                        doc_id=top_chunk.doc_id,
                        doc_language=doc_lang,
                        answerability_score=round(ans_score, 4),
                        refusal_reason=str(ans_result.reason),
                    )
                    latency.guardrail_post_ms = round(post_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=ans_result.reason,
                        refusal_message=ans_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Stage: Language Consistency Guardrail (Prevents language mismatch)
            if self.language_consistency:
                t0 = time.perf_counter()
                lang_result = self.language_consistency.validate(
                    query=query,
                    evidence_text=top_chunk.text,
                    query_language=query_lang,
                    evidence_language=doc_lang,
                    fallback_used=fallback_used,
                )
                post_guard_ms += (time.perf_counter() - t0) * 1000
                if not lang_result.passed:
                    logger.info(
                        "language_consistency_guard_refusal",
                        trace_id=trace_id,
                        query=query[:100],
                        query_language=query_lang,
                        doc_language=doc_lang,
                        fallback_used=fallback_used,
                        refusal_reason=str(lang_result.reason),
                        message=lang_result.message[:100],
                    )
                    latency.guardrail_post_ms = round(post_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=lang_result.reason,
                        refusal_message=lang_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            context = retrieval_result.context_text

            # Stage: Pre-generation guardrail (context coverage check)
            if self.coverage:
                t0 = time.perf_counter()
                coverage_result = self.coverage.check(query, context, semantic_score=top_score)
                post_guard_ms += (time.perf_counter() - t0) * 1000
                if not coverage_result.passed:
                    latency.guardrail_post_ms = round(post_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=coverage_result.reason,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Diagnostic Telemetry Logging — Full Visibility into Language-Aware Retrieval
            candidate_lang_dist = {}
            top_k_details = []
            for idx, c in enumerate(retrieval_result.chunks):
                l = c.metadata.get("language") or detect_language(c.text)
                candidate_lang_dist[l] = candidate_lang_dist.get(l, 0) + 1
                top_k_details.append({
                    "rank": idx + 1,
                    "doc_id": c.doc_id[:50] if c.doc_id else "unknown",
                    "language": l,
                    "is_same_language": (l == query_lang),
                    "dense_score": round(c.score, 4),
                    "rerank_score": round(c.metadata.get("rerank_score", c.score), 4),
                    "answerability_score": round(c.metadata.get("answerability_score", 1.0), 4),
                })

            # Comprehensive diagnostic log for retrieval debugging
            try:
                logger.info(
                    "rag_pipeline_language_aware_retrieval_diagnostics",
                    trace_id=trace_id,
                    query=query[:100],
                    detected_query_language=query_lang,
                    raw_stt_language=input_language or "not_provided",
                    candidate_language_distribution=candidate_lang_dist,
                    total_candidates_pool=len(retrieval_result.chunks),
                    fallback_used=fallback_used,
                    final_selected_document={
                        "doc_id": top_chunk.doc_id,
                        "language": doc_lang,
                        "query_language_match": (doc_lang == query_lang),
                        "dense_score": round(top_score, 4),
                        "rerank_score": round(top_chunk.metadata.get("rerank_score", top_score), 4),
                        "answerability_score": round(top_chunk.metadata.get("answerability_score", 1.0), 4),
                    },
                    top_k_candidates_detail=top_k_details,
                    guardrail_checks={
                        "grounding_threshold": grounding_threshold,
                        "grounding_score": round(top_score, 4),
                        "grounding_passed": top_score >= grounding_threshold,
                        "answerability_passed": top_chunk.metadata.get("answerability_score", 1.0) >= 0.40,
                        "language_consistency_passed": (doc_lang == query_lang),
                        "language_consistency_allow_fallback": fallback_used,
                    },
                    final_decision="GROUNDED_ANSWER",
                )
            except Exception:
                pass

            # Stage: Generate
            if settings.answer_mode.strip().lower() == "fast":
                answer = extractive_answer(query, retrieval_result.chunks)
                if not answer:
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=RefusalReason.UNGROUNDED,
                        refusal_message="No grounded source sentence was found for this question.",
                        trace_id=trace_id,
                        latency=latency,
                    )

                # Fast mode still validates the final answer. Extraction is
                # source-only, but source language and question answerability
                # must be checked after sentence selection as well.
                t0 = time.perf_counter()
                answer_language = detect_language(answer)
                answer_language_result = self.language_consistency.validate(
                    query=query,
                    evidence_text=answer,
                    query_language=query_lang,
                    evidence_language=answer_language,
                )
                post_guard_ms += (time.perf_counter() - t0) * 1000
                if not answer_language_result.passed:
                    latency.guardrail_post_ms = round(post_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=answer_language_result.reason,
                        refusal_message=answer_language_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

                t0 = time.perf_counter()
                final_answerability = self.answerability.validate(
                    query, answer, query_language=query_lang
                )
                post_guard_ms += (time.perf_counter() - t0) * 1000
                if not final_answerability.passed:
                    latency.guardrail_post_ms = round(post_guard_ms, 2)
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=final_answerability.reason,
                        refusal_message=final_answerability.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

                if self.grounding:
                    post_start = time.perf_counter()
                    final_grounding = self.grounding.check(answer, context)
                    post_guard_ms += (time.perf_counter() - post_start) * 1000
                    if not final_grounding.passed:
                        latency.guardrail_post_ms = round(post_guard_ms, 2)
                        latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                        return PipelineResult(
                            query=query,
                            success=False,
                            refused=True,
                            refusal_reason=final_grounding.reason,
                            refusal_message=final_grounding.message,
                            trace_id=trace_id,
                            latency=latency,
                        )

                latency.guardrail_post_ms = round(post_guard_ms, 2)
                latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                return PipelineResult(
                    answer=answer,
                    query=query,
                    success=True,
                    trace_id=trace_id,
                    latency=latency,
                    retrieved_chunks=[
                        {"text": c.text, "score": c.score, "doc_id": c.doc_id, "language": c.metadata.get("language")}
                        for c in retrieval_result.chunks
                    ],
                )

            try:
                gen_result = await with_retry(
                    self.llm_primary.generate, query, context, max_retries=1
                )
            except Exception:
                # Fallback to secondary LLM
                if self.llm_fallback:
                    logger.warning("llm_primary_failed_using_fallback", trace_id=trace_id)
                    gen_result = await self.llm_fallback.generate(query, context)
                else:
                    raise

            latency.generation_ms = gen_result.duration_ms

            # Stage: Grounding check
            post_start = time.perf_counter()
            if self.grounding:
                grounding_result = self.grounding.check(gen_result.answer, context)
                latency.guardrail_post_ms = round((time.perf_counter() - post_start) * 1000, 2)

                if not grounding_result.passed:
                    latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
                    return PipelineResult(
                        query=query,
                        success=False,
                        refused=True,
                        refusal_reason=grounding_result.reason,
                        refusal_message=grounding_result.message,
                        trace_id=trace_id,
                        latency=latency,
                    )

            # Success
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)

            result = PipelineResult(
                answer=gen_result.answer,
                query=query,
                success=True,
                model_used=gen_result.model,
                is_fallback=gen_result.is_fallback,
                trace_id=trace_id,
                latency=latency,
                retrieved_chunks=[
                    {"text": c.text, "score": c.score, "doc_id": c.doc_id}
                    for c in retrieval_result.chunks
                ],
            )

            logger.info(
                "pipeline_complete",
                trace_id=trace_id,
                total_ms=latency.total_ms,
                model=gen_result.model,
                chunks_used=len(retrieval_result.chunks),
            )

            return result

        except Exception as e:
            latency.total_ms = round((time.perf_counter() - pipeline_start) * 1000, 2)
            logger.error("pipeline_text_error", trace_id=trace_id, error=str(e))
            return PipelineResult(
                query=query,
                success=False,
                refused=True,
                refusal_reason=RefusalReason.SYSTEM_ERROR,
                refusal_message="An internal error occurred while processing your question.",
                trace_id=trace_id,
                latency=latency,
            )
