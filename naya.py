"""
RAG backend for the Masters Union UG Data Science & AI course assistant.

Architecture:
  Data Sources  ->  Chunker  ->  ChromaDB + BM25 (Hybrid)
       |
  Query Expansion (HyDE) -> Hybrid Retrieval -> RRF Re-ranking
       |
  Confidence Check -> Groq LLM (streaming) -> Answer + Follow-ups + Source
"""

import os
import re

import chromadb
from dotenv import load_dotenv
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── Load environment ──────────────────────────────────────────────────────────

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

_DIR = os.path.dirname(os.path.abspath(__file__))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not set. Create a .env file in the project root or export it."
    )

GROQ_MODEL = "llama-3.1-8b-instant"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "mu_dsai_v3"

DATA_SOURCES = {
    "brochure":   os.path.join(_DIR, "DSAI_Brochure_Structured.txt"),
    "webpage":    os.path.join(_DIR, "scraped_web_data.txt"),
    "fallback":   os.path.join(_DIR, "program_data.txt"),
    "additional": os.path.join(_DIR, "additionaldata.txt"),
}

# ── Query topic classification ─────────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "fees":       ["fee", "cost", "price", "tuition", "inr", "lakh", "scholarship",
                   "discount", "payment", "afford", "expensive", "cheap", "admission fee",
                   "emi", "loan", "financial aid"],
    "curriculum": ["curriculum", "course", "subject", "learn", "teach", "topic",
                   "year 1", "year 2", "year 3", "year 4", "semester", "module",
                   "python", "ml", "ai", "data science", "deep learning", "nlp",
                   "project", "outclass", "syllabus", "capstone", "genai"],
    "admissions": ["admit", "admission", "apply", "eligible", "eligibility", "jee",
                   "sat", "musat", "aptitude", "test", "essay", "video", "criteria",
                   "qualify", "selection", "when can i join", "intake", "deadline"],
    "career":     ["job", "placement", "career", "salary", "hire", "recruiter",
                   "internship", "work", "company", "startup", "entrepreneur",
                   "after graduation", "what can i do", "ctc", "lpa", "package"],
    "global":     ["international", "global", "abroad", "usa", "chicago", "illinois",
                   "silicon valley", "germany", "japan", "singapore", "dubai",
                   "immersion", "foreign", "us degree", "dual degree", "3+1"],
    "campus":     ["campus", "hostel", "dorm", "location", "gurugram", "dlf",
                   "gym", "cafeteria", "facility", "club", "life", "residential",
                   "maker lab", "library"],
    "faculty":    ["faculty", "professor", "teacher", "instructor", "mentor",
                   "who teaches", "speaker", "expert", "coach", "industry expert"],
    "contact":    ["contact", "email", "phone", "reach", "address", "call",
                   "office", "ugadmissions"],
}


def classify_query(question: str) -> str:
    """Return the most likely topic for a user question."""
    lower = question.lower()
    scores = {topic: 0 for topic in TOPIC_KEYWORDS}
    for topic, kws in TOPIC_KEYWORDS.items():
        for kw in kws:
            if kw in lower:
                scores[topic] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# ── Comparison detection ──────────────────────────────────────────────────────

_COMPARE_PATTERNS = [
    r"compare\s+(.+?)\s+(?:vs\.?|versus|and|with|to)\s+(.+)",
    r"difference\s+between\s+(.+?)\s+and\s+(.+)",
    r"(.+?)\s+vs\.?\s+(.+)",
]


def detect_comparison(question: str) -> tuple[bool, str, str]:
    """Detect if query is a comparison. Returns (is_comparison, aspect_a, aspect_b)."""
    lower = question.lower().strip().rstrip("?")
    for pattern in _COMPARE_PATTERNS:
        match = re.search(pattern, lower)
        if match:
            return True, match.group(1).strip(), match.group(2).strip()
    return False, "", ""


# ── Global singletons ──────────────────────────────────────────────────────────

_embed_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_groq: Groq | None = None
_initialized = False

# BM25 parallel index
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[str] = []
_bm25_ids: list[str] = []
_bm25_metas: list[dict] = []


def initialize():
    global _embed_model, _collection, _groq, _initialized
    if _initialized:
        return

    print("[init] Loading embedding model ...")
    _embed_model = SentenceTransformer(EMBED_MODEL)

    print("[init] Connecting to Groq ...")
    _groq = Groq(api_key=GROQ_API_KEY)

    print("[init] Setting up ChromaDB ...")
    chroma = chromadb.Client()
    _collection = chroma.get_or_create_collection(COLLECTION)

    if _collection.count() == 0:
        print("[init] Collection empty -- indexing data sources ...")
        _index_all_sources()
    else:
        print(f"[init] Collection has {_collection.count()} chunks -- skipping re-index")

    _initialized = True
    print("[init] Ready.")


# ── BM25 tokenizer ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ── Indexing ───────────────────────────────────────────────────────────────────

def _load_source(label: str, path: str) -> str:
    if not os.path.exists(path):
        print(f"[index] {label}: file not found at {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"[index] {label}: loaded {len(text)} chars from {os.path.basename(path)}")
    return text


def _detect_section(chunk: str) -> str:
    """Heuristic: detect the section tag from a chunk of text."""
    lower = chunk.lower()
    if any(kw in lower for kw in ["fee", "inr", "scholarship", "tuition", "lakh", "₹"]):
        return "fees"
    if any(kw in lower for kw in ["year 1", "year 2", "year 3", "year 4",
                                   "curriculum", "course", "python", "machine learning",
                                   "deep learning", "nlp", "mlops", "capstone",
                                   "syllabus", "module"]):
        return "curriculum"
    if any(kw in lower for kw in ["admission", "apply", "jee", "sat", "musat",
                                   "aptitude", "video essay", "eligibility"]):
        return "admissions"
    if any(kw in lower for kw in ["placement", "career", "internship", "job",
                                   "recruiter", "startup", "entrepreneur",
                                   "ctc", "lpa", "package", "salary"]):
        return "career"
    if any(kw in lower for kw in ["illinois tech", "chicago", "silicon valley",
                                   "global immersion", "germany", "japan", "singapore",
                                   "dual degree", "3+1"]):
        return "global"
    if any(kw in lower for kw in ["campus", "hostel", "gurugram", "dlf", "gym",
                                   "cafeteria", "club", "maker lab"]):
        return "campus"
    if any(kw in lower for kw in ["faculty", "professor", "cto", "ph.d", "mentor",
                                   "speaker", "google", "microsoft", "nasa",
                                   "industry expert"]):
        return "faculty"
    if any(kw in lower for kw in ["contact", "ugadmissions", "+91", "phone", "email",
                                   "address"]):
        return "contact"
    if any(kw in lower for kw in ["class profile", "cohort", "batch", "diversity",
                                   "student profile", "background"]):
        return "class_profile"
    return "general"


def _index_all_sources():
    global _bm25_index, _bm25_corpus, _bm25_ids, _bm25_metas

    # 800 chars / 160 overlap = 20% overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
    doc_ids: list[str] = []
    docs: list[str] = []
    embeds: list[list] = []
    metas: list[dict] = []

    chunk_idx = 0

    for label, path in DATA_SOURCES.items():
        text = _load_source(label, path)
        if not text:
            continue
        chunks = splitter.split_text(text)
        print(f"[index] {label}: {len(chunks)} chunks")
        for chunk in chunks:
            section = _detect_section(chunk)
            embedding = _embed_model.encode(chunk).tolist()
            doc_ids.append(f"{label}_{chunk_idx}")
            docs.append(chunk)
            embeds.append(embedding)
            metas.append({"source": label, "section": section})
            chunk_idx += 1

    _collection.add(
        ids=doc_ids,
        documents=docs,
        embeddings=embeds,
        metadatas=metas,
    )

    # Build BM25 index in parallel
    tokenized_corpus = [_tokenize(doc) for doc in docs]
    _bm25_index = BM25Okapi(tokenized_corpus)
    _bm25_corpus = docs
    _bm25_ids = doc_ids
    _bm25_metas = metas

    print(f"[index] Total indexed: {chunk_idx} chunks (ChromaDB + BM25)")


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    semantic_results: list[tuple[str, str, dict, float]],
    bm25_results: list[tuple[str, str, dict, float]],
    k: int = 60,
    n: int = 7,
) -> list[tuple[str, str, dict, float]]:
    """
    Merge two ranked lists using RRF.
    RRF score for doc d = sum over all rankers R of: 1 / (k + rank_R(d))
    Returns top-n (doc_id, doc_text, metadata, rrf_score) tuples.
    """
    scores: dict = {}

    for rank, (doc_id, doc, meta, _) in enumerate(semantic_results):
        if doc_id not in scores:
            scores[doc_id] = {"doc": doc, "meta": meta, "score": 0.0}
        scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    for rank, (doc_id, doc, meta, _) in enumerate(bm25_results):
        if doc_id not in scores:
            scores[doc_id] = {"doc": doc, "meta": meta, "score": 0.0}
        scores[doc_id]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        ("", item["doc"], item["meta"], item["score"])
        for item in ranked[:n]
    ]


# ── HyDE Query Expansion ─────────────────────────────────────────────────────

_HYDE_PROMPT = """You are a query expander for a university programme chatbot.
Given the user's question about the Masters' Union UG Data Science & AI programme,
generate 2-3 alternative search queries that would help find the answer.
Return ONLY the queries, one per line. No numbering, no explanation.

User question: {question}

Alternative search queries:"""


def _expand_query(question: str) -> list[str]:
    """Use LLM to generate 2-3 alternative phrasings. Returns original + expansions."""
    try:
        response = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": _HYDE_PROMPT.format(question=question)}],
            temperature=0.7,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        expansions = [line.strip() for line in raw.splitlines() if line.strip()]
        return [question] + expansions[:3]
    except Exception:
        return [question]


# ── Retrieval (Hybrid: BM25 + Semantic + RRF) ────────────────────────────────

def _retrieve(
    question: str,
    topic: str,
    n: int = 7,
    use_hyde: bool = True,
) -> tuple[str, list[dict], list[float]]:
    """
    Hybrid retrieval: BM25 + semantic search, merged via RRF.
    Optionally expands the query using HyDE.
    Returns (context_text, list_of_metadata, list_of_rrf_scores).
    """
    queries = _expand_query(question) if use_hyde else [question]

    all_semantic: list[tuple[str, str, dict, float]] = []
    all_bm25: list[tuple[str, str, dict, float]] = []

    for q in queries:
        embedding = _embed_model.encode(q).tolist()

        # Semantic search via ChromaDB
        try:
            sem_results = _collection.query(
                query_embeddings=[embedding],
                n_results=n,
                where={"section": topic} if topic != "general" else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # Fallback if topic filter returns too few
            sem_results = _collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )

        for i in range(len(sem_results["ids"][0])):
            all_semantic.append((
                sem_results["ids"][0][i],
                sem_results["documents"][0][i],
                sem_results["metadatas"][0][i],
                sem_results["distances"][0][i],
            ))

        # Also do unfiltered semantic search to broaden coverage
        try:
            sem_broad = _collection.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            for i in range(len(sem_broad["ids"][0])):
                all_semantic.append((
                    sem_broad["ids"][0][i],
                    sem_broad["documents"][0][i],
                    sem_broad["metadatas"][0][i],
                    sem_broad["distances"][0][i],
                ))
        except Exception:
            pass

        # BM25 search
        if _bm25_index is not None:
            tokens = _tokenize(q)
            bm25_scores = _bm25_index.get_scores(tokens)
            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True,
            )[:n * 2]
            for idx in top_indices:
                if bm25_scores[idx] > 0:
                    all_bm25.append((
                        _bm25_ids[idx],
                        _bm25_corpus[idx],
                        _bm25_metas[idx],
                        bm25_scores[idx],
                    ))

    # Merge via RRF
    fused = _reciprocal_rank_fusion(all_semantic, all_bm25, n=n)

    docs = [item[1] for item in fused]
    metas = [item[2] for item in fused]
    rrf_scores = [item[3] for item in fused]

    context = "\n\n---\n\n".join(docs)
    return context, metas, rrf_scores


# ── Confidence assessment ─────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.015


def _assess_confidence(rrf_scores: list[float]) -> tuple[str, float]:
    """
    Returns (confidence_level, best_score).
    confidence_level is 'high', 'medium', or 'low'.
    """
    if not rrf_scores:
        return "low", 0.0
    best = max(rrf_scores)
    avg_top3 = sum(rrf_scores[:3]) / min(3, len(rrf_scores))

    if best >= 0.030 and avg_top3 >= 0.020:
        return "high", best
    elif best >= CONFIDENCE_THRESHOLD:
        return "medium", best
    else:
        return "low", best


# ── Follow-up parsing ─────────────────────────────────────────────────────────

def parse_followups(answer: str) -> tuple[str, list[str]]:
    """Split the LLM response into the main answer and follow-up questions."""
    lines = answer.split("\n")
    main_lines = []
    followups = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("FOLLOWUP:"):
            followups.append(stripped.replace("FOLLOWUP:", "").strip())
        else:
            main_lines.append(line)
    main_answer = "\n".join(main_lines).rstrip()
    return main_answer, followups


# ── Answer generation ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a friendly and knowledgeable admissions assistant for the \
Masters' Union Undergraduate Programme in Data Science & AI.

Rules:
1. Answer ONLY using the CONTEXT provided below. Never guess or use outside knowledge.
2. If the answer is not in the context, say exactly:
   "This information isn't available in our programme materials. \
Please contact admissions: ugadmissions@mastersunion.org | +91-7669186660"
3. Be concise and structured. Use bullet points for lists.
4. Always cite which source (Brochure / Webpage) the information comes from if clear.
5. Do not add follow-up questions to the response."""

_USER_PROMPT = """\
CONTEXT:
{context}

QUESTION: {question}

Answer:"""

_COMPARE_SYSTEM_PROMPT = """You are a knowledgeable admissions assistant for the \
Masters' Union UG Data Science & AI programme.

The user wants to compare two aspects of the programme. You are given context for BOTH topics.
Present your answer as a clear side-by-side comparison using a structured format.
Use a markdown table or two clearly labeled sections. Be factual and cite sources.
At the END, add 2-3 follow-up questions prefixed with "FOLLOWUP:" on separate lines."""

_COMPARE_USER_PROMPT = """\
CONTEXT FOR "{aspect_a}":
{context_a}

CONTEXT FOR "{aspect_b}":
{context_b}

QUESTION: {question}

Provide a clear comparison:"""


def ask_question(question: str) -> dict:
    """
    Main RAG pipeline.

    Returns dict with: answer, source, topic, confidence, followups
    """
    initialize()

    topic = classify_query(question)
    context, meta_list, rrf_scores = _retrieve(question, topic)

    confidence, best_score = _assess_confidence(rrf_scores)

    if confidence == "low":
        answer = (
            "I couldn't find this information in the programme documents. "
            "This question may be outside the scope of what I know about the "
            "Masters' Union UG Data Science & AI programme.\n\n"
            "For specific queries, please contact admissions:\n"
            "- Email: ugadmissions@mastersunion.org\n"
            "- Phone: +91-7669186660"
        )
        return {
            "answer": answer,
            "source": "",
            "topic": topic,
            "confidence": confidence,
            "followups": [
                "What is the fee structure for the programme?",
                "What subjects are covered in the curriculum?",
                "How do I apply for admission?",
            ],
        }

    source_labels = sorted({m["source"] + " / " + m["section"] for m in meta_list})
    source_note = "Sources: " + ", ".join(source_labels)
    display_source = f"{source_note}\n\n{context}"

    prompt = _USER_PROMPT.format(context=context, question=question)

    system_prompt = _SYSTEM_PROMPT
    if confidence == "medium":
        system_prompt += (
            "\n6. Your confidence in the retrieved context is moderate. "
            "If you're unsure about any detail, say so and recommend contacting admissions."
        )

    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )

    raw_answer = response.choices[0].message.content.strip()
    main_answer, followups = parse_followups(raw_answer)

    return {
        "answer": main_answer,
        "source": display_source,
        "topic": topic,
        "confidence": confidence,
        "followups": followups,
    }


def ask_question_stream(question: str) -> dict:
    """
    Streaming version of ask_question.
    Returns dict where 'answer_stream' is a generator (or None for low confidence).
    """
    initialize()

    topic = classify_query(question)
    context, meta_list, rrf_scores = _retrieve(question, topic)

    confidence, best_score = _assess_confidence(rrf_scores)

    if confidence == "low":
        return {
            "answer_stream": None,
            "answer_static": (
                "I couldn't find this information in the programme documents. "
                "This question may be outside the scope of what I know about the "
                "Masters' Union UG Data Science & AI programme.\n\n"
                "For specific queries, please contact admissions:\n"
                "- Email: ugadmissions@mastersunion.org\n"
                "- Phone: +91-7669186660"
            ),
            "source": "",
            "topic": topic,
            "confidence": confidence,
            "followups": [
                "What is the fee structure for the programme?",
                "What subjects are covered in the curriculum?",
                "How do I apply for admission?",
            ],
        }

    source_labels = sorted({m["source"] + " / " + m["section"] for m in meta_list})
    source_note = "Sources: " + ", ".join(source_labels)
    display_source = f"{source_note}\n\n{context}"

    prompt = _USER_PROMPT.format(context=context, question=question)

    system_prompt = _SYSTEM_PROMPT
    if confidence == "medium":
        system_prompt += (
            "\n6. Your confidence in the retrieved context is moderate. "
            "If you're unsure about any detail, say so and recommend contacting admissions."
        )

    stream = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
        stream=True,
    )

    def token_generator():
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return {
        "answer_stream": token_generator(),
        "answer_static": None,
        "source": display_source,
        "topic": topic,
        "confidence": confidence,
        "followups": [],  # will be parsed after streaming completes
    }


def ask_comparison(question: str, aspect_a: str, aspect_b: str) -> dict:
    """Handle comparison queries with dual retrieval."""
    initialize()

    topic_a = classify_query(aspect_a)
    topic_b = classify_query(aspect_b)

    context_a, metas_a, scores_a = _retrieve(aspect_a, topic_a, n=5)
    context_b, metas_b, scores_b = _retrieve(aspect_b, topic_b, n=5)

    prompt = _COMPARE_USER_PROMPT.format(
        aspect_a=aspect_a, context_a=context_a,
        aspect_b=aspect_b, context_b=context_b,
        question=question,
    )

    response = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )

    raw_answer = response.choices[0].message.content.strip()
    main_answer, followups = parse_followups(raw_answer)

    src_a = sorted({m["source"] + "/" + m["section"] for m in metas_a})
    src_b = sorted({m["source"] + "/" + m["section"] for m in metas_b})
    combined_source = (
        f"[{aspect_a}] Sources: {', '.join(src_a)}\n\n"
        f"[{aspect_b}] Sources: {', '.join(src_b)}\n\n"
        f"--- Context for {aspect_a} ---\n{context_a}\n\n"
        f"--- Context for {aspect_b} ---\n{context_b}"
    )

    return {
        "answer": main_answer,
        "source": combined_source,
        "topic": "comparison",
        "confidence": "high",
        "followups": followups,
    }


def ask_comparison_stream(question: str, aspect_a: str, aspect_b: str) -> dict:
    """Streaming version of ask_comparison."""
    initialize()

    topic_a = classify_query(aspect_a)
    topic_b = classify_query(aspect_b)

    context_a, metas_a, scores_a = _retrieve(aspect_a, topic_a, n=5)
    context_b, metas_b, scores_b = _retrieve(aspect_b, topic_b, n=5)

    prompt = _COMPARE_USER_PROMPT.format(
        aspect_a=aspect_a, context_a=context_a,
        aspect_b=aspect_b, context_b=context_b,
        question=question,
    )

    src_a = sorted({m["source"] + "/" + m["section"] for m in metas_a})
    src_b = sorted({m["source"] + "/" + m["section"] for m in metas_b})
    combined_source = (
        f"[{aspect_a}] Sources: {', '.join(src_a)}\n\n"
        f"[{aspect_b}] Sources: {', '.join(src_b)}\n\n"
        f"--- Context for {aspect_a} ---\n{context_a}\n\n"
        f"--- Context for {aspect_b} ---\n{context_b}"
    )

    stream = _groq.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
        stream=True,
    )

    def token_generator():
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    return {
        "answer_stream": token_generator(),
        "answer_static": None,
        "source": combined_source,
        "topic": "comparison",
        "confidence": "high",
        "followups": [],
    }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_questions = [
        "What is the fee structure?",
        "Am I eligible without a JEE score?",
        "What topics are covered in Year 2?",
        "What jobs can I get after this programme?",
        "Is there a hostel facility?",
        "How do I apply?",
        "Tell me about PwC partnership",
    ]
    for q in test_questions:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        result = ask_question(q)
        print(f"A: {result['answer']}")
        print(f"[Topic: {result['topic']} | Confidence: {result['confidence']}]")
        if result["followups"]:
            print(f"Follow-ups: {result['followups']}")
