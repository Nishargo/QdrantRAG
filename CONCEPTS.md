# Technical Concepts — RAG Hybrid Search Engine

A plain-language explanation of every algorithm and concept used in this project.

---

## 1. Chunking (Recursive Word Chunking)

A PDF can have hundreds of pages. Embedding models have a size limit — you
cannot feed the entire document at once. So we split the text into small
overlapping pieces called **chunks**.

**How it works:**
- Take the full text and split into individual words
- Take N words at a time (chunkSize = 40)
- Slide forward by (chunkSize - overlap) words = 35
- Repeat until end of document

**Why overlap?**
If an important sentence sits at the boundary between two chunks, the overlap
captures it in both — so no meaning gets lost at the edges.
```
Text:    [word1 word2 word3 ... word40 word41 word42 ... word80]
Chunk 1: [word1 .................. word40]
Chunk 2:                  [word36 .................. word75]
                          ↑ 5-word overlap
```

---

## 2. Embeddings (Vector Representations)

An embedding converts text into a list of numbers (a vector) that represents
its meaning mathematically.

**Example (simplified to 3 dimensions):**
```
"I love dogs"           → [0.91, 0.23, 0.45]
"Dogs are my favorite"  → [0.89, 0.21, 0.47]  ← similar meaning, similar numbers
"Stock market crashed"  → [0.12, 0.87, 0.03]  ← different meaning, different numbers
```

In this project we use `nomic-embed-text` via Ollama which produces
**768-dimensional vectors** — 768 numbers per chunk of text, running
100% locally on your machine.

---

## 3. Vector Similarity (Cosine Similarity)

Once we have vectors, we measure how similar two are using **cosine similarity**
— the angle between two vectors in 768-dimensional space.
```
Small angle  → vectors point in similar direction → similar meaning → score ≈ 1.0
Large angle  → vectors point in opposite directions → different meaning → score ≈ 0.0
```

**Why cosine and not subtraction?**
Cosine ignores the length of the vector and only measures direction. This makes
it robust when texts have different lengths — a short chunk and a long chunk
about the same topic still score high similarity.

---

## 4. BM25 (Best Match 25)

BM25 is a keyword search algorithm from the 1990s still used by Elasticsearch
and Google today. It scores documents based on exact word matches.

**Two core ideas:**

**IDF — Inverse Document Frequency**
Rare words score higher than common words. If "retrieval" appears in only 2 out
of 86 chunks, it's a strong signal. If "the" appears in all 86 chunks, it tells
us nothing.
```
IDF = log((N - df + 0.5) / (df + 0.5) + 1)

N  = total chunks (86)
df = chunks containing this term
```

**TF with Length Normalization**
A short chunk where "RAG" is the main topic scores higher than a long chunk
that merely mentions "RAG" once. BM25 adjusts for document length:
```
TF_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunkLen / avgLen))

k1 = 1.5  (term frequency saturation — diminishing returns on repetition)
b  = 0.75 (length normalization strength)
```

---

## 5. Hybrid Search

Neither vector search nor BM25 is perfect alone:

| | Vector Search | BM25 |
|---|---|---|
| Strength | Meaning, synonyms, paraphrasing | Exact keywords, acronyms, codes |
| Weakness | Exact rare terms like "SP-API" | Synonyms — "fever" misses "elevated temperature" |

**Hybrid search combines both** — you get semantic understanding AND exact
keyword matching in one result set.

---

## 6. Reciprocal Rank Fusion (RRF)

After running both searches we have two separate ranked lists. RRF merges them
fairly using this formula:
```
score = 1 / (60 + rank)
```

Each result gets a score from each list based on its rank position:
- Rank 1 → 1/61 = 0.01639
- Rank 2 → 1/62 = 0.01613
- Rank 5 → 1/65 = 0.01538

Results that appear in **both** lists receive scores from both and float to the
top. That is why results marked `Vector + BM25 ⭐` are the highest confidence
— two independent systems agreed.

**Why 60?** It is a constant that prevents rank 1 from dominating too strongly.
It was established in the original RRF research paper (Cormack et al. 2009) and
has become the standard default.

---

## 7. Qdrant

Qdrant is a **vector database** — a database designed specifically to store and
search vectors efficiently. Unlike a normal database (which matches exact values),
Qdrant finds approximate nearest neighbors in high-dimensional space using HNSW
(Hierarchical Navigable Small World) graphs.

Each stored item (called a **point**) has:
- `id` — unique identifier
- `vector` — the 768 numbers representing meaning
- `payload` — the original text and any metadata

---

## References

- BM25: Robertson & Zaragoza (2009) — "The Probabilistic Relevance Framework"
- RRF: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank Fusion outperforms Condorcet"
- nomic-embed-text: https://huggingface.co/nomic-ai/nomic-embed-text-v1
- Qdrant docs: https://qdrant.tech/documentation