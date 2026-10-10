"""LangChain + LangSmith RAG Recall Evaluation Pipeline.

Evaluates retrieval quality (Recall@3, Recall@5, Recall@10) across evaluation questions
from the LangSmith dataset against the ground-truth Transformer PDF source.

Usage:
    python evaluate_recall.py --mode inspect
    python evaluate_recall.py --mode prepare-labels
    python evaluate_recall.py --mode review-labels
    python evaluate_recall.py --mode evaluate
    python evaluate_recall.py --mode test-formula
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure standard UTF-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RecallEvaluation")

# Default paths and settings
DEFAULT_PDF_PATH = os.getenv("EVAL_PDF_PATH", "app/data/uploads/lec19.transformer.pdf")
DEFAULT_DATASET_NAME = os.getenv("EVAL_DATASET_NAME", "rag")
DEFAULT_OUTPUT_DIR = Path("evaluation_results")
DEFAULT_PARENT_CHUNK_SIZE = 1000
DEFAULT_PARENT_CHUNK_OVERLAP = 100
DEFAULT_K_VALUES = [3, 5, 10]


# ============================================================================
# 1. Mathematical Recall Formula & Validation
# ============================================================================

def calculate_recall_at_k(
    relevant_ids: Set[str],
    retrieved_ids: List[str],
    k: int,
) -> Optional[float]:
    """Calculate Recall@k as a percentage using the exact formula:
    
        Recall@k (%) = 100 * |R ∩ D_k| / |R|
        
    Where:
        R is the set of verified ground-truth relevant chunk IDs for the question.
        D_k is the set of unique chunk IDs returned in the top-k retrieval results.
        |R ∩ D_k| is the number of relevant chunks successfully retrieved.
        |R| is the total number of known relevant chunks.
        
    Returns:
        float percentage between 0.0 and 100.0, or None if R is empty (unlabeled).
    """
    if not relevant_ids or len(relevant_ids) == 0:
        return None  # Unlabeled or no verified ground truth

    # Extract unique top-k retrieved IDs
    top_k_retrieved = retrieved_ids[:k]
    unique_retrieved_k = set(top_k_retrieved)

    # Compute set intersection |R ∩ D_k|
    matched_ids = relevant_ids.intersection(unique_retrieved_k)
    recall_fraction = len(matched_ids) / len(relevant_ids)
    
    return round(recall_fraction * 100.0, 2)


def test_recall_formula_synthetic() -> bool:
    """Run mathematical unit verification on synthetic examples and edge cases."""
    print("\n" + "=" * 70)
    print("RUNNING SYNTHETIC RECALL FORMULA TEST SUITE")
    print("=" * 70)
    
    test_cases = [
        # (Name, R, D_k, k, Expected Recall)
        ("Prompt Example: 2 of 4 matched at k=5", {"A", "B", "C", "D"}, ["A", "C", "X", "Y", "Z"], 5, 50.0),
        ("Full Recall (100%): 2 of 2 matched at k=3", {"A", "B"}, ["A", "B", "C"], 3, 100.0),
        ("Zero Recall (0%): 0 of 2 matched at k=3", {"A", "B"}, ["X", "Y", "Z"], 3, 0.0),
        ("Top-k Truncation: Matched item is at rank 4, evaluating k=3", {"A", "B"}, ["X", "Y", "Z", "A"], 3, 0.0),
        ("Top-k Included: Matched item is at rank 4, evaluating k=5", {"A", "B"}, ["X", "Y", "Z", "A", "W"], 5, 50.0),
        ("Duplicate IDs in retrieval deduplicated", {"A", "B"}, ["A", "A", "X", "Y"], 4, 50.0),
        ("Single relevant chunk retrieved at k=1", {"CHUNK_1"}, ["CHUNK_1", "CHUNK_2"], 1, 100.0),
        ("Unlabeled ground truth (empty R)", set(), ["A", "B", "C"], 3, None),
    ]

    all_passed = True
    for name, r_set, d_list, k_val, expected in test_cases:
        actual = calculate_recall_at_k(r_set, d_list, k_val)
        passed = (actual == expected) if expected is not None else (actual is None)
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"[{status}] {name}")
        print(f"   Ground Truth R = {r_set}")
        print(f"   Retrieved (k={k_val}) = {d_list[:k_val]}")
        print(f"   Expected: {expected} | Computed: {actual}%\n")

    if all_passed:
        print("All synthetic mathematical unit tests PASSED successfully!")
    else:
        print("ERROR: One or more mathematical test cases failed.")
    print("=" * 70 + "\n")
    return all_passed


# ============================================================================
# 2. Source Document Loading & Stable Chunk Inventory
# ============================================================================

def generate_stable_chunk_id(filename: str, page_number: int, chunk_index: int) -> str:
    """Generate a clean, deterministic, unique chunk ID."""
    clean_fn = Path(filename).stem.replace(".", "_").replace(" ", "_").replace("-", "_")
    return f"{clean_fn}_p{page_number}_c{chunk_index}"


def build_source_chunk_inventory(
    pdf_path: str = DEFAULT_PDF_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = DEFAULT_PARENT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_PARENT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Load PDF, split into deterministic chunks, and save complete source inventory."""
    resolved_path = Path(pdf_path)
    if not resolved_path.is_absolute():
        resolved_path = (PROJECT_ROOT / pdf_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Transformer PDF not found at: {resolved_path}")

    logger.info("Loading Transformer PDF from: %s", resolved_path)
    
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFLoader(str(resolved_path))
    pages = loader.load()
    logger.info("Successfully loaded %d pages from PDF.", len(pages))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    inventory: List[Dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_file = output_dir / "source_chunks.jsonl"

    for page_idx, page in enumerate(pages, 1):
        raw_text = page.page_content.strip()
        if not raw_text:
            continue

        page_chunks = splitter.split_text(raw_text)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            chunk_id = generate_stable_chunk_id(resolved_path.name, page_idx, chunk_idx)
            record = {
                "chunk_id": chunk_id,
                "source": resolved_path.name,
                "page": page_idx,
                "chunk_index": chunk_idx,
                "text": chunk_text,
                "char_length": len(chunk_text),
            }
            inventory.append(record)

    with open(inventory_file, "w", encoding="utf-8") as f:
        for item in inventory:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Saved complete inventory of %d source chunks to %s", len(inventory), inventory_file)
    return inventory


# ============================================================================
# 3. Ground-Truth Passage Annotations for 20 Evaluation Questions
# ============================================================================

# High-precision verified ground-truth passages mapped directly to lec19.transformer.pdf
VERIFIED_GROUND_TRUTH_PASSAGES: Dict[str, Dict[str, Any]] = {
    "What problem does the Transformer architecture address?": {
        "pages": [2, 3, 4, 6],
        "rationale": "Pages 2-4 and 6 introduce the motivation for the Transformer, discussing limitations of recurrent/convolutional networks, universal multimodal representation, parallel processing, and replacing recurrence with self-attention.",
        "keywords": ["transformer architecture", "attention is all you need", "handling all modality", "scaling", "recurrence"],
    },
    "What is the role of multi-head attention?": {
        "pages": [35, 36, 37, 38],
        "rationale": "Pages 35-38 define multi-head attention, explaining how queries, keys, and values are projected into multiple subspaces with distinct heads in parallel and concatenated to capture diverse representation subspaces.",
        "keywords": ["multi-head attention", "head", "projection", "subspace", "concatenate"],
    },
    "Why are residual connections used in Transformer layers?": {
        "pages": [32, 33, 34],
        "rationale": "Pages 32-34 explain Add & Norm with residual connections (x + SubLayer(x)), which facilitate gradient flow and prevent vanishing gradients in deep encoder and decoder stacks.",
        "keywords": ["residual connection", "add & norm", "layer normalization", "sublayer"],
    },
    "What is an attention head?": {
        "pages": [35, 36],
        "rationale": "Pages 35-36 explicitly define an attention head as an individual learned linear projection of queries, keys, and values that computes scaled dot-product attention independently.",
        "keywords": ["attention head", "linear projection", "head_i", "scaled dot-product"],
    },
    "What is the difference between an encoder and a decoder in the original Transformer?": {
        "pages": [39, 40, 46, 47, 48],
        "rationale": "Pages 39-40 and 46-48 detail the structural distinction: the encoder processes input representations using bidirectional self-attention, while the decoder generates target tokens autoregressively using masked self-attention and cross-attention over encoder outputs.",
        "keywords": ["encoder", "decoder", "masked attention", "cross attention", "autoregressive"],
    },
    "Does the original Transformer use positional encoding in both encoder and decoder inputs?": {
        "pages": [25, 26, 40, 46],
        "rationale": "Pages 25-26, 40, and 46 confirm that positional encoding vectors are added to input token embeddings at the base of both the encoder stack and the decoder stack.",
        "keywords": ["position encoding", "positional encoding", "encoder input", "decoder input"],
    },
    "What is the purpose of masking in decoder self-attention?": {
        "pages": [43, 44, 45, 46],
        "rationale": "Pages 43-46 explain causal triangular masking in decoder self-attention, which sets future position scores to -infinity so predictions at position i depend only on known outputs at positions < i.",
        "keywords": ["masking", "masked self-attention", "causal mask", "subsequent positions", "-inf"],
    },
    "Does the original Transformer require recurrence to process sequences?": {
        "pages": [6, 8, 9],
        "rationale": "Pages 6, 8, and 9 contrast attention with RNNs, showing the Transformer eliminates recurrent connections in favor of parallel self-attention operations.",
        "keywords": ["recurrence", "rnn", "seq2seq", "parallel", "attention is all you need"],
    },
    "What is the difference between self-attention and encoder-decoder attention?": {
        "pages": [9, 47, 48],
        "rationale": "Pages 9 and 47-48 contrast self-attention (where Q, K, V all come from the same sequence) with encoder-decoder cross-attention (where Q comes from decoder states and K, V come from encoder output memory).",
        "keywords": ["self-attention", "cross-attention", "encoder-decoder", "queries from decoder", "keys and values from encoder"],
    },
    "Why does the Transformer use scaled dot-product attention?": {
        "pages": [21, 22, 23],
        "rationale": "Pages 21-23 explain that dividing query-key dot products by sqrt(d_k) prevents dot products from growing large for large dimensions, which would otherwise push softmax into regions with extremely small gradients.",
        "keywords": ["scaled dot-product", "scale", "sqrt(d_k)", "dimension", "vanishing gradient", "softmax"],
    },
    "How does the original Transformer decoder prevent access to future target tokens?": {
        "pages": [43, 44, 45, 46],
        "rationale": "Pages 43-46 illustrate masked self-attention using a lower-triangular mask that zeros out / sets to -inf attention weights for future token indices.",
        "keywords": ["decoder", "future target tokens", "mask", "autoregressive generation"],
    },
    "What are the main components of the Transformer encoder layer?": {
        "pages": [7, 39, 40],
        "rationale": "Pages 7, 39, and 40 list the sublayers of an encoder layer: Multi-Head Self-Attention, Position-wise Feed-Forward Network, Residual Connections (Add), and Layer Normalization (Norm).",
        "keywords": ["encoder layer", "multi-head attention", "feed-forward", "add & norm", "layer norm"],
    },
    "Why does the original Transformer scale dot products by the square root of the key dimension?": {
        "pages": [21, 22, 23],
        "rationale": "Pages 21-23 provide the mathematical explanation that for large vector dimension d_k, dot products grow large in magnitude, causing softmax to yield extremely peaked distributions with near-zero gradients.",
        "keywords": ["square root", "key dimension", "sqrt(d_k)", "magnitude", "gradient"],
    },
    "What is the role of the position-wise feed-forward network?": {
        "pages": [29, 30, 31],
        "rationale": "Pages 29-31 describe the position-wise Feed-Forward Network (FFN), consisting of two linear transformations with a non-linear activation (ReLU/GELU) applied identically to each position.",
        "keywords": ["feed-forward", "ffn", "position-wise", "linear", "relu", "two linear transformations"],
    },
    "What is the main advantage of attention over recurrence in the original Transformer?": {
        "pages": [3, 4, 6, 8, 9],
        "rationale": "Pages 3-4, 6, and 8-9 highlight that attention allows fully parallelizable training across all sequence positions and creates constant O(1) path lengths between any pair of input tokens.",
        "keywords": ["advantage", "parallel computation", "constant path length", "scaling", "no recurrence"],
    },
    "How does the Transformer process tokens in parallel during training?": {
        "pages": [6, 18, 19, 20, 44],
        "rationale": "Pages 6, 18-20, and 44 demonstrate that all queries, keys, and values for the entire sequence are packed into matrices (Q, K, V), allowing simultaneous matrix multiplications during training.",
        "keywords": ["parallel", "matrix multiplication", "training", "q k v matrices", "simultaneous"],
    },
    "What are queries, keys, and values in attention?": {
        "pages": [9, 16, 17, 18],
        "rationale": "Pages 9 and 16-18 define queries (what a token is looking for), keys (what a token contains / addresses), and values (the actual content / representation vectors to be retrieved).",
        "keywords": ["query", "key", "value", "q", "k", "v", "linear transformation"],
    },
    "What does the softmax function do in scaled dot-product attention?": {
        "pages": [19, 20, 21],
        "rationale": "Pages 19-21 show that softmax takes raw scaled dot-product logits and normalizes them into non-negative attention weights that sum to 1.0 across the sequence.",
        "keywords": ["softmax", "attention weights", "probability", "sum to 1", "normalize"],
    },
    "What is the purpose of positional encoding?": {
        "pages": [25, 26, 27, 28],
        "rationale": "Pages 25-28 explain that because self-attention is permutation-equivariant and does not inherently know sequence order, positional encodings are injected to represent relative or absolute token positions.",
        "keywords": ["position encoding", "order", "sequence order", "sinusoidal", "permutation"],
    },
    "What is self-attention in a Transformer?": {
        "pages": [9, 18, 19, 20],
        "rationale": "Pages 9 and 18-20 explain self-attention as the mechanism where an input sequence attends to itself, generating contextual representations by computing attention weights across all tokens in the sequence.",
        "keywords": ["self-attention", "query key value", "sequence attends to itself", "contextual representation"],
    },
}


def prepare_ground_truth_labels(
    dataset_name: str = DEFAULT_DATASET_NAME,
    inventory: List[Dict[str, Any]] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> List[Dict[str, Any]]:
    """Build and save verified ground-truth labels for all evaluation questions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_file = output_dir / "ground_truth.jsonl"

    # Index source chunks by page number
    if inventory is None:
        inventory = build_source_chunk_inventory(output_dir=output_dir)

    chunks_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for c in inventory:
        p = c["page"]
        chunks_by_page.setdefault(p, []).append(c)

    # Fetch questions from LangSmith
    dataset_examples: List[Dict[str, Any]] = []
    try:
        from langsmith import Client
        client = Client()
        examples = list(client.list_examples(dataset_name=dataset_name))
        logger.info("Retrieved %d examples from LangSmith dataset '%s'.", len(examples), dataset_name)
        for ex in examples:
            q = (ex.inputs.get("question") or ex.inputs.get("input") or "").strip()
            a = (ex.outputs.get("answer") or ex.outputs.get("output") or "").strip() if ex.outputs else ""
            dataset_examples.append({
                "example_id": str(ex.id),
                "question": q,
                "reference_answer": a,
            })
    except Exception as err:
        logger.warning("Could not connect to LangSmith to fetch dataset '%s': %s", dataset_name, err)
        logger.info("Using embedded verified evaluation questions suite.")
        for q_text, p_info in VERIFIED_GROUND_TRUTH_PASSAGES.items():
            dataset_examples.append({
                "example_id": f"offline_{len(dataset_examples)+1}",
                "question": q_text,
                "reference_answer": p_info.get("rationale", ""),
            })

    ground_truth_records: List[Dict[str, Any]] = []

    for idx, ex in enumerate(dataset_examples, 1):
        q = ex["question"]
        ref_a = ex["reference_answer"]
        q_clean = q.strip().rstrip("?")
        
        # Match with verified passages
        matched_info = None
        for key, val in VERIFIED_GROUND_TRUTH_PASSAGES.items():
            if key.lower().strip() == q.lower().strip() or key.lower().startswith(q_clean.lower()[:30]):
                matched_info = val
                break

        relevant_chunk_ids: List[str] = []
        relevant_pages: List[int] = []
        relevant_chunks_detail: List[Dict[str, Any]] = []
        rationale = ""
        label_status = "unverified"

        if matched_info:
            relevant_pages = matched_info["pages"]
            rationale = matched_info["rationale"]
            label_status = "verified"

            for page_num in relevant_pages:
                page_chunks = chunks_by_page.get(page_num, [])
                for chk in page_chunks:
                    relevant_chunk_ids.append(chk["chunk_id"])
                    relevant_chunks_detail.append({
                        "chunk_id": chk["chunk_id"],
                        "page": page_num,
                        "snippet": chk["text"][:150],
                    })
        else:
            rationale = "Candidate passage search required for this query."
            label_status = "needs_review"

        gt_record = {
            "question_id": ex["example_id"],
            "question_index": idx,
            "question": q,
            "reference_answer": ref_a,
            "relevant_chunk_ids": list(dict.fromkeys(relevant_chunk_ids)),
            "relevant_pages": relevant_pages,
            "label_status": label_status,
            "rationale": rationale,
            "relevant_chunks_detail": relevant_chunks_detail,
        }
        ground_truth_records.append(gt_record)

    with open(gt_file, "w", encoding="utf-8") as f:
        for rec in ground_truth_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("Saved ground-truth annotations for %d questions to %s", len(ground_truth_records), gt_file)
    return ground_truth_records


# ============================================================================
# 4. Actual Retrieval Execution Adapter
# ============================================================================

def retrieve_chunks_for_question(
    question: str,
    top_k: int = 10,
    inventory_map: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Execute actual retrieval using existing VectorStore and map results to stable chunk IDs."""
    from app.vectorstore.croma_store import get_vectorstore
    
    vs = get_vectorstore()
    raw_results = vs.similarity_search_with_score(question, k=top_k)
    
    retrieved_records: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for rank, (doc, distance) in enumerate(raw_results, 1):
        metadata = dict(doc.metadata or {})
        page = metadata.get("page", 1)
        source = metadata.get("filename", metadata.get("source", "lec19.transformer.pdf"))
        
        # Resolve stable chunk ID
        # Format: lec19_transformer_p{page}_c0
        stable_id = generate_stable_chunk_id(source, page, 0)
        
        # Deduplicate identical chunk IDs in top-k
        if stable_id in seen_ids:
            continue
        seen_ids.add(stable_id)

        # Convert Euclidean/L2 distance to cosine/similarity score
        similarity_score = round(1.0 / (1.0 + float(distance)), 4)

        retrieved_records.append({
            "rank": rank,
            "chunk_id": stable_id,
            "page": page,
            "source": source,
            "score": similarity_score,
            "text": doc.page_content,
        })

    return retrieved_records


# ============================================================================
# 5. Core Evaluation Engine & LangSmith Traceability
# ============================================================================

def run_recall_evaluation(
    dataset_name: str = DEFAULT_DATASET_NAME,
    pdf_path: str = DEFAULT_PDF_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    k_values: List[int] = DEFAULT_K_VALUES,
    log_to_langsmith: bool = True,
    experiment_name: str | None = None,
) -> Dict[str, Any]:
    """Run full Recall@k evaluation against all dataset questions and log to LangSmith."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Source Inventory
    inventory = build_source_chunk_inventory(pdf_path=pdf_path, output_dir=output_dir)
    inventory_map = {item["chunk_id"]: item for item in inventory}

    # 2. Ground-Truth Labels
    ground_truth_records = prepare_ground_truth_labels(
        dataset_name=dataset_name,
        inventory=inventory,
        output_dir=output_dir,
    )

    # 3. LangSmith Client Setup
    ls_client = None
    if log_to_langsmith:
        try:
            from app.tracing.langsmith import setup_tracing
            setup_tracing()
            from langsmith import Client
            ls_client = Client()
            logger.info("LangSmith client initialized successfully.")
        except Exception as ls_err:
            logger.warning("LangSmith initialization skipped or offline: %s", ls_err)
            ls_client = None

    per_question_results: List[Dict[str, Any]] = []
    all_retrieved_logs: List[Dict[str, Any]] = []

    max_k = max(k_values)
    timestamp_str = datetime.now(timezone.utc).isoformat()
    exp_name = experiment_name or f"recall-eval-{int(time.time())}"

    # Track metrics
    valid_labeled_count = 0
    macro_recall_sums: Dict[int, float] = {k: 0.0 for k in k_values}
    micro_matched_counts: Dict[int, int] = {k: 0 for k in k_values}
    total_ground_truth_chunks = 0

    print("\n" + "=" * 80)
    print(f"STARTING RECALL EVALUATION ON DATASET: '{dataset_name}' (Total Questions: {len(ground_truth_records)})")
    print(f"Evaluated k values: {k_values} | Ground-Truth Source: {Path(pdf_path).name}")
    print("=" * 80 + "\n")

    for q_data in ground_truth_records:
        q_id = q_data["question_id"]
        q_idx = q_data["question_index"]
        question = q_data["question"]
        ref_answer = q_data["reference_answer"]
        relevant_ids = set(q_data["relevant_chunk_ids"])
        label_status = q_data["label_status"]

        # Execute Retrieval
        retrieved_chunks = retrieve_chunks_for_question(question, top_k=max_k, inventory_map=inventory_map)
        retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]

        # Calculate Recall for each k
        recalls_at_k: Dict[str, Optional[float]] = {}
        matched_ids_at_k: Dict[str, List[str]] = {}
        missed_ids_at_k: Dict[str, List[str]] = {}

        is_valid_labeled = len(relevant_ids) > 0 and label_status == "verified"

        if is_valid_labeled:
            valid_labeled_count += 1
            total_ground_truth_chunks += len(relevant_ids)

        for k in k_values:
            k_key = f"recall_at_{k}"
            k_pct = calculate_recall_at_k(relevant_ids, retrieved_ids, k)
            recalls_at_k[k_key] = k_pct

            top_k_set = set(retrieved_ids[:k])
            matched = list(relevant_ids.intersection(top_k_set))
            missed = list(relevant_ids - top_k_set)

            matched_ids_at_k[f"matched_at_{k}"] = matched
            missed_ids_at_k[f"missed_at_{k}"] = missed

            if is_valid_labeled and k_pct is not None:
                macro_recall_sums[k] += k_pct
                micro_matched_counts[k] += len(matched)

        # Log retrieved chunks detailed record
        for chk in retrieved_chunks:
            matches_gt = chk["chunk_id"] in relevant_ids
            all_retrieved_logs.append({
                "question_id": q_id,
                "question": question,
                "rank": chk["rank"],
                "chunk_id": chk["chunk_id"],
                "page": chk["page"],
                "similarity_score": chk["score"],
                "matches_ground_truth": matches_gt,
                "snippet": chk["text"][:150],
            })

        q_result = {
            "question_id": q_id,
            "question_index": q_idx,
            "question": question,
            "reference_answer": ref_answer,
            "label_status": label_status,
            "relevant_chunk_ids": list(relevant_ids),
            "relevant_chunk_count": len(relevant_ids),
            "retrieved_chunk_ids": retrieved_ids,
            "recalls_at_k": recalls_at_k,
            "matched_ids_at_k": matched_ids_at_k,
            "missed_ids_at_k": missed_ids_at_k,
            "rationale": q_data.get("rationale", ""),
            "relevant_pages": q_data.get("relevant_pages", []),
        }
        per_question_results.append(q_result)

        # Terminal output per question
        r3_str = f"{recalls_at_k.get('recall_at_3', 'N/A')}%" if recalls_at_k.get('recall_at_3') is not None else "N/A"
        r5_str = f"{recalls_at_k.get('recall_at_5', 'N/A')}%" if recalls_at_k.get('recall_at_5') is not None else "N/A"
        r10_str = f"{recalls_at_k.get('recall_at_10', 'N/A')}%" if recalls_at_k.get('recall_at_10') is not None else "N/A"

        print(f"Q{q_idx:02d}: {question}")
        print(f"     Ground Truth Relevant Chunks ({len(relevant_ids)}): {list(relevant_ids)}")
        print(f"     Retrieved Top-5 IDs: {retrieved_ids[:5]}")
        print(f"     Recall@3: {r3_str} | Recall@5: {r5_str} | Recall@10: {r10_str}")
        print(f"     Matched at k=5: {matched_ids_at_k.get('matched_at_5')} | Missed: {missed_ids_at_k.get('missed_at_5')}\n")

        # 4. Log individual evaluation trace to LangSmith
        if ls_client is not None:
            try:
                ls_client.create_run(
                    name=f"RecallEval: Q{q_idx}",
                    run_type="chain",
                    inputs={"question": question, "reference_answer": ref_answer},
                    outputs={
                        "recalls_at_k": recalls_at_k,
                        "matched_at_5": matched_ids_at_k.get("matched_at_5"),
                        "missed_at_5": missed_ids_at_k.get("missed_at_5"),
                        "retrieved_ids": retrieved_ids,
                    },
                    extra={
                        "metadata": {
                            "experiment_name": exp_name,
                            "dataset_name": dataset_name,
                            "question_id": q_id,
                            "relevant_chunk_ids": list(relevant_ids),
                            "recall_at_3_percent": recalls_at_k.get("recall_at_3"),
                            "recall_at_5_percent": recalls_at_k.get("recall_at_5"),
                            "recall_at_10_percent": recalls_at_k.get("recall_at_10"),
                        },
                    },
                    tags=["evaluation", "recall", f"dataset:{dataset_name}", f"exp:{exp_name}"],
                )
            except Exception as trace_err:
                logger.debug("Failed logging individual LangSmith run: %s", trace_err)

    # Calculate Overall Macro & Micro Metrics
    macro_overall: Dict[str, float] = {}
    micro_overall: Dict[str, float] = {}

    for k in k_values:
        if valid_labeled_count > 0:
            macro_val = round(macro_recall_sums[k] / valid_labeled_count, 2)
            macro_overall[f"overall_macro_recall_at_{k}"] = macro_val
            
            if total_ground_truth_chunks > 0:
                micro_val = round((micro_matched_counts[k] / total_ground_truth_chunks) * 100.0, 2)
            else:
                micro_val = 0.0
            micro_overall[f"overall_micro_recall_at_{k}"] = micro_val
        else:
            macro_overall[f"overall_macro_recall_at_{k}"] = 0.0
            micro_overall[f"overall_micro_recall_at_{k}"] = 0.0

    summary_report = {
        "timestamp": timestamp_str,
        "experiment_name": exp_name,
        "dataset_name": dataset_name,
        "source_pdf": Path(pdf_path).name,
        "total_questions": len(ground_truth_records),
        "valid_labeled_questions": valid_labeled_count,
        "unlabeled_or_review_needed": len(ground_truth_records) - valid_labeled_count,
        "total_ground_truth_chunks": total_ground_truth_chunks,
        "macro_recall_percentages": macro_overall,
        "micro_recall_percentages": micro_overall,
        "per_question_results": per_question_results,
    }

    # Save outputs
    summary_file = output_dir / "recall_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2, ensure_ascii=False)

    retrieved_file = output_dir / "retrieved_chunks.jsonl"
    with open(retrieved_file, "w", encoding="utf-8") as f:
        for item in all_retrieved_logs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Final Terminal Report
    print("=" * 80)
    print("RECALL EVALUATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Dataset Questions:         {len(ground_truth_records)}")
    print(f"Questions Evaluated:             {len(ground_truth_records)}")
    print(f"Questions with Verified Labels:  {valid_labeled_count}")
    print(f"Questions Requiring Review:      {len(ground_truth_records) - valid_labeled_count}")
    print(f"Total Ground-Truth Chunks:       {total_ground_truth_chunks}")
    print("-" * 80)
    print("MACRO-AVERAGED RECALL SCORES:")
    for k in k_values:
        print(f"  • Overall Macro Recall@{k:02d}:   {macro_overall.get(f'overall_macro_recall_at_{k}', 0.0):.2f}%")
    print("-" * 80)
    print("MICRO-AVERAGED RECALL SCORES:")
    for k in k_values:
        print(f"  • Overall Micro Recall@{k:02d}:   {micro_overall.get(f'overall_micro_recall_at_{k}', 0.0):.2f}%")
    print("=" * 80)
    print(f"Results saved to:\n  1. {summary_file}\n  2. {retrieved_file}\n  3. {output_dir / 'source_chunks.jsonl'}\n  4. {output_dir / 'ground_truth.jsonl'}")
    print("=" * 80 + "\n")

    return summary_report


# ============================================================================
# 6. Interactive / CLI Modes Dispatcher
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LangChain + LangSmith RAG Recall Evaluation Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["inspect", "prepare-labels", "review-labels", "evaluate", "test-formula"],
        default="evaluate",
        help=(
            "Execution mode:\n"
            "  inspect        - Load PDF and generate source_chunks.jsonl inventory\n"
            "  prepare-labels - Generate ground_truth.jsonl annotations\n"
            "  review-labels  - Review and inspect ground-truth passage rationales\n"
            "  evaluate       - Run full Recall@k evaluation & log to LangSmith\n"
            "  test-formula   - Run synthetic mathematical unit test suite\n"
        ),
    )
    parser.add_argument("--pdf-path", default=DEFAULT_PDF_PATH, help="Path to ground-truth Transformer PDF")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="LangSmith dataset name")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for results")
    parser.add_argument("--k-values", nargs="+", type=int, default=DEFAULT_K_VALUES, help="List of k values (e.g. 3 5 10)")
    parser.add_argument("--no-langsmith", action="store_true", help="Disable LangSmith API logging")
    parser.add_argument("--experiment-name", default=None, help="Custom LangSmith experiment name")

    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    if args.mode == "test-formula":
        passed = test_recall_formula_synthetic()
        sys.exit(0 if passed else 1)

    elif args.mode == "inspect":
        inventory = build_source_chunk_inventory(pdf_path=args.pdf_path, output_dir=out_dir)
        print(f"\nPDF Inspection complete. Total {len(inventory)} chunks written to {out_dir / 'source_chunks.jsonl'}")

    elif args.mode == "prepare-labels":
        inventory = build_source_chunk_inventory(pdf_path=args.pdf_path, output_dir=out_dir)
        labels = prepare_ground_truth_labels(dataset_name=args.dataset_name, inventory=inventory, output_dir=out_dir)
        print(f"\nGround-truth labels generated for {len(labels)} questions in {out_dir / 'ground_truth.jsonl'}")

    elif args.mode == "review-labels":
        gt_file = out_dir / "ground_truth.jsonl"
        if not gt_file.exists():
            print("ground_truth.jsonl not found. Preparing labels first...")
            labels = prepare_ground_truth_labels(dataset_name=args.dataset_name, output_dir=out_dir)
        else:
            with open(gt_file, "r", encoding="utf-8") as f:
                labels = [json.loads(line) for line in f if line.strip()]

        print("\n" + "=" * 80)
        print(f"GROUND-TRUTH LABELS REVIEW ({len(labels)} Questions)")
        print("=" * 80)
        for item in labels:
            print(f"Q{item['question_index']:02d} [{item['label_status'].upper()}]: {item['question']}")
            print(f"   Relevant Chunks ({len(item['relevant_chunk_ids'])}): {item['relevant_chunk_ids']}")
            print(f"   Source Pages: {item.get('relevant_pages')}")
            print(f"   Rationale: {item.get('rationale')}\n")
        print("=" * 80)

    elif args.mode == "evaluate":
        run_recall_evaluation(
            dataset_name=args.dataset_name,
            pdf_path=args.pdf_path,
            output_dir=out_dir,
            k_values=args.k_values,
            log_to_langsmith=not args.no_langsmith,
            experiment_name=args.experiment_name,
        )


if __name__ == "__main__":
    main()
