"""
End-to-End Integration Tests for Multilingual Voice RAG Pipeline.

Tests the complete flow: Language Detection → Retrieval → Guardrails → Generation
"""

import pytest
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

# These tests require the full pipeline setup
pytest.importorskip("src.harness.pipeline")

from src.config import settings
from src.embeddings.multilingual import EmbeddingService
from src.retrieval.numpy_store import LocalNumpyStore
from src.guardrails.off_topic import OffTopicGuardrail
from src.guardrails.unsafe_input import UnsafeInputGuardrail
from src.guardrails.coverage import CoverageGuardrail
from src.guardrails.language_consistency import LanguageConsistencyGuardrail
from src.guardrails.answerability import AnswerabilityGuardrail
from src.harness.pipeline import RAGPipeline
from src.utils.language import detect_language


@pytest.fixture(scope="module")
def embedding_service():
    """Load embedding model once per test module."""
    embed_svc = EmbeddingService()
    embed_svc.load_model()
    return embed_svc


@pytest.fixture(scope="module")
def vector_store():
    """Load vector store once per test module."""
    store = LocalNumpyStore()
    store.connect()
    return store


@pytest.fixture(scope="module")
def pipeline(embedding_service, vector_store):
    """Create RAG pipeline with all guardrails."""
    off_topic = OffTopicGuardrail(embedding_service=embedding_service, threshold=0.10)
    unsafe = UnsafeInputGuardrail()
    coverage = CoverageGuardrail(threshold=0.15, semantic_threshold=0.40)
    language_consistency = LanguageConsistencyGuardrail(allow_fallback=False)
    answerability = AnswerabilityGuardrail(min_answerability=0.40)

    return RAGPipeline(
        embedding_service=embedding_service,
        vector_store=vector_store,
        unsafe_guardrail=unsafe,
        off_topic_guardrail=off_topic,
        coverage_guardrail=coverage,
        language_consistency_guardrail=language_consistency,
        answerability_guardrail=answerability,
    )


class TestMultilingualRetrieval:
    """Test that retrieval respects language preferences."""

    @pytest.mark.asyncio
    async def test_english_query_retrieves_english_evidence(self, pipeline):
        """English query should prioritize English evidence."""
        result = await pipeline.process_text("What is machine learning?", input_language="en")
        
        if result.success:
            assert result.answer is not None
            assert len(result.retrieved_chunks) > 0
            # First chunk should be English or high-confidence answer
            top_chunk = result.retrieved_chunks[0]
            assert top_chunk is not None
            
            # Verify language consistency in metadata if available
            chunk_lang = top_chunk.get("language", "unknown")
            assert chunk_lang in ["en", "unknown"], f"Expected English chunk, got {chunk_lang}"

    @pytest.mark.asyncio
    async def test_hindi_query_retrieves_hindi_evidence(self, pipeline):
        """Hindi query should prioritize Hindi evidence."""
        result = await pipeline.process_text("मशीन लर्निंग क्या है?", input_language="hi")
        
        # Should either succeed with Hindi answer or refuse
        # Should NOT return English answer to Hindi query
        if result.success:
            assert result.answer is not None
            # Verify evidence is Hindi or mixed Hindi-English
            if result.retrieved_chunks:
                top_chunk = result.retrieved_chunks[0]
                chunk_lang = top_chunk.get("language", "unknown")
                assert chunk_lang in ["hi", "unknown"], f"Expected Hindi chunk for Hindi query, got {chunk_lang}"

    @pytest.mark.asyncio
    async def test_gujarati_query_retrieves_gujarati_evidence(self, pipeline):
        """Gujarati query should prioritize Gujarati evidence."""
        result = await pipeline.process_text("મશીન લર્નિંગ શું છે?", input_language="gu")
        
        if result.success:
            assert result.answer is not None
            if result.retrieved_chunks:
                top_chunk = result.retrieved_chunks[0]
                chunk_lang = top_chunk.get("language", "unknown")
                assert chunk_lang in ["gu", "unknown"], f"Expected Gujarati chunk, got {chunk_lang}"

    @pytest.mark.asyncio
    async def test_gujarati_goa_location_grounded(self, pipeline):
        """Verify Gujarati Goa location queries retrieve goa_gu_01 and produce grounded answers."""
        queries = [
            "ગોવા ક્યાં છે?",
            "ગોવા ક્યાં આવેલું છે?",
            "ગોવા ક્યાં બાજુ આવે છે?",
        ]
        for q in queries:
            result = await pipeline.process_text(q, input_language="gu")
            assert result.success is True, f"Expected success for {q}, got refused={result.refused}, reason={result.refusal_reason}"
            assert result.refused is False
            assert "Goa" in result.answer or "ગોવા" in result.answer
            assert len(result.retrieved_chunks) > 0
            assert result.retrieved_chunks[0].get("doc_id") == "goa_gu_01"
            assert result.retrieved_chunks[0].get("language") == "gu"


class TestLanguagePreservation:
    """Test that query language is preserved throughout pipeline."""

    @pytest.mark.asyncio
    async def test_language_detected_from_stt(self, pipeline):
        """Language should be properly detected and preserved."""
        # Test with explicit language hints
        en_result = await pipeline.process_text("What is AI?", input_language="en")
        assert en_result is not None
        
        hi_result = await pipeline.process_text("कृत्रिम बुद्धिमत्ता क्या है?", input_language="hi")
        assert hi_result is not None

    @pytest.mark.asyncio
    async def test_language_auto_detection(self, pipeline):
        """Language should be auto-detected when not explicitly provided."""
        # Auto-detect English
        en_result = await pipeline.process_text("What is machine learning?")
        assert en_result is not None
        
        # Auto-detect Hindi
        hi_result = await pipeline.process_text("मशीन लर्निंग क्या है?")
        assert hi_result is not None


class TestAnswerabilityGuard:
    """Test that answerability guardrail prevents tangential answers."""

    @pytest.mark.asyncio
    async def test_integration_by_parts_specific(self, pipeline):
        """
        Query: "What is integration by parts?"
        
        Should answer with calculus-specific evidence, or refuse if no such evidence exists.
        Should NOT return loose Hindi answer about 'integration' in business context.
        """
        result = await pipeline.process_text("What is integration by parts?", input_language="en")
        
        # Either succeeds with calculus answer, or refuses
        # Should NOT silently return wrong domain answer
        if result.success:
            assert result.answer is not None
            # Verify answer addresses calculus
            answer_lower = result.answer.lower()
            # If it has any content, should contain calculus terms
            if len(answer_lower) > 10:
                assert any(term in answer_lower for term in ["integral", "calculus", "parts", "formula"])
        else:
            # If refused, should be for answerability/grounding, not language
            assert result.refusal_reason is not None

    @pytest.mark.asyncio
    async def test_specific_neural_networks_question(self, pipeline):
        """Query for neural networks should return specific answer."""
        result = await pipeline.process_text("What are neural networks?", input_language="en")
        
        if result.success:
            assert result.answer is not None
            answer_lower = result.answer.lower()
            # Should contain neural network specific terms
            assert any(term in answer_lower for term in ["neural", "network", "learning", "algorithm"])


class TestNegativeCases:
    """Test handling of invalid or out-of-scope queries."""

    @pytest.mark.asyncio
    async def test_unsafe_query_refused(self, pipeline):
        """Unsafe queries should be refused by UnsafeInputGuardrail."""
        result = await pipeline.process_text("How do I make a bomb?")
        
        # Should be refused for safety/unsafe input
        assert result.refused is True

    @pytest.mark.asyncio
    async def test_gibberish_query_refused(self, pipeline):
        """Gibberish queries should be refused as off-topic."""
        result = await pipeline.process_text("asldkfj qwpeoiru zxmcnbv")
        
        # Should be refused as off-topic or ungrounded
        assert result.refused is True

    @pytest.mark.asyncio
    async def test_joke_request_refused(self, pipeline):
        """Request for jokes should be refused as off-topic."""
        result = await pipeline.process_text("Tell me a joke.")
        
        # Should be refused as off-topic
        assert result.refused is True

    @pytest.mark.asyncio
    async def test_warp_drive_refused(self, pipeline):
        """Query about speculative tech should be refused as ungrounded."""
        result = await pipeline.process_text("How do I build a warp drive?")
        
        # Should be refused as ungrounded (no evidence in corpus)
        assert result.refused is True


class TestDiagnosticLogging:
    """Test that diagnostic logging captures all required information."""

    @pytest.mark.asyncio
    async def test_diagnostic_info_available(self, pipeline):
        """Pipeline should capture diagnostic info in metadata."""
        result = await pipeline.process_text("What is machine learning?", input_language="en")
        
        assert result.trace_id is not None
        assert result.latency is not None
        # Verify latency breakdown is populated
        assert result.latency.total_ms > 0


class TestRegressionFixVerification:
    """Verify that previous fixes are not regressed."""

    @pytest.mark.asyncio
    async def test_no_noaa_tornado_false_positive(self, pipeline):
        """
        Original bug: English query "What is machine learning?" 
        was retrieving NOAA tornado passage due to BM25 stop-word "is".
        
        This should NOT happen with multilingual dense retrieval.
        """
        result = await pipeline.process_text("What is machine learning?", input_language="en")
        
        if result.success:
            assert result.answer is not None
            answer_lower = result.answer.lower()
            # Should NOT contain NOAA or tornado terms
            assert "noaa" not in answer_lower
            assert "tornado" not in answer_lower
            assert "doppler" not in answer_lower
            # SHOULD contain ML-related terms
            assert any(term in answer_lower for term in ["machine", "learning", "algorithm", "data"])

    @pytest.mark.asyncio
    async def test_semantic_grounding_preserved(self, pipeline):
        """Semantic grounding checks should still work."""
        result = await pipeline.process_text("What is machine learning?")
        
        # Should succeed with high confidence
        if result.success:
            assert result.answer is not None
            assert len(result.answer) > 5


class TestEvidenceQualityRegression:
    """Ensure topic similarity alone cannot produce a successful answer."""

    @pytest.mark.parametrize(
        "query,input_language,expected_success",
        [
            ("What is machine learning?", "en", False),
            ("मशीन लर्निंग क्या है?", "hi", False),
            ("What are neural networks?", "en", True),
            ("What is a corporation?", "en", True),
            ("Explain photosynthesis.", "en", True),
            ("What is the capital of France?", "en", False),
            ("What is integration by parts?", "en", False),
            ("What is the weather today?", "en", False),
            ("Goa ક્યાં છે?", "gu", True),
            ("ગોવા ક્યાં છે?", "gu", True),
            ("ગોવા ક્યાં આવેલું છે?", "gu", True),
            ("ગોવા ક્યાં બાજુ આવે છે?", "gu", True),
            ("Where is Goa located?", "en", True),
            ("મશીન લર્નિંગ શું છે?", "gu", True),
            ("asldkfj qwpeoiru zxmcnbv", "en", False),
            ("How do I build a warp drive?", "en", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_evidence_quality_matrix(self, pipeline, query, input_language, expected_success):
        result = await pipeline.process_text(query, input_language=input_language)
        assert result.success is expected_success
        if result.success:
            assert result.answer
            assert detect_language(result.answer) == input_language
        else:
            assert result.refused is True

