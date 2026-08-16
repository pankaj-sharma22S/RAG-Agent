"""Retriever package providing Hybrid Retrieval (Dense BGE-M3 + BM25 Sparse), RRF, Cross-Encoder, CRAG, and Query Refinement."""

from .crag import CRAGEvaluator, global_crag_evaluator
from .dense_retriever import DenseRetriever, global_dense_retriever
from .hybrid_retriever import HybridRetriever, global_hybrid_retriever
from .query_refiner import QueryRefiner, global_query_refiner
from .reranker import CrossEncoderReranker, global_reranker
from .sparse_retriever import SparseRetriever, global_sparse_retriever

__all__ = [
    "DenseRetriever",
    "global_dense_retriever",
    "SparseRetriever",
    "global_sparse_retriever",
    "HybridRetriever",
    "global_hybrid_retriever",
    "CrossEncoderReranker",
    "global_reranker",
    "CRAGEvaluator",
    "global_crag_evaluator",
    "QueryRefiner",
    "global_query_refiner",
]
