# 🎬 What Actually Makes Movies "Similar"?

`A content-based movie recommender that started as a metadata matcher, broke on its own test cases, and evolved into a plot-embedding system with a documented trail of every failure that shaped it.`

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Python](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Library: sentence-transformers](https://img.shields.io/badge/Library-sentence--transformers-orange.svg)](https://www.sbert.net/)
[![Library: Pandas](https://img.shields.io/badge/Library-Pandas-150458.svg)](https://pandas.pydata.org/)
[![API: OMDb](https://img.shields.io/badge/API-OMDb-red.svg)](https://www.omdbapi.com/)
[![Notebook: Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange.svg)](https://jupyter.org/)

## 📜 Project Overview

Most "movie recommender" tutorials skip a real design question: **what does "similar" actually mean, and does your data support answering it?**

This project started with a 7,668-film IMDb dataset (1980–2020) that has one genre, one director, one writer, one lead actor per film — no plot, no keywords. The obvious approach was a metadata recommender: match movies by shared director/genre/cast. It technically worked, and it was wrong in an instructive way — querying *The Shining* returned *Full Metal Jacket* near the top, purely because they share a director. Two Kubrick films with nothing else in common. That's the moment this project's actual question became clear: **production credits are not a reliable proxy for "this feels like that movie."**

The fix was to go get the thing the original dataset never had — plot text — enrich it via the OMDb API, and build similarity on sentence embeddings instead of shared credits. That pivot introduced its own failures, each of which is documented below rather than quietly fixed and hidden. The point of this README isn't just "here's a working recommender" — it's "here's the evidence for why it works the way it does."

## ✨ Key Findings

- **Metadata-only matching is genre-blind to tone.** Same director ≠ same kind of movie (*The Shining* → *Full Metal Jacket* was the case that killed this approach).
- **Short plot summaries can be too thin to compare.** *The Godfather* and *Goodfellas* — arguably the two most iconic mafia films ever made — scored a similarity of only **0.234** on short OMDb plots, because the two summaries emphasized completely different narrative beats and shared almost no vocabulary.
- **More text isn't automatically better.** Switching to OMDb's full-length plots fixed some thin-vocabulary cases but *measurably hurt others* — re-querying *The Shining* on full plots dropped its top match score from 0.60 to 0.50 and let tonally unrelated films (a conspiracy thriller, a comedy) into the top 5. Longer, multi-topic plot text dilutes the specific signal that a tight, genre-defining one-liner concentrated.
- **Plot similarity misses franchise/sequel relationships you'd expect it to catch.** *The Godfather* and *The Godfather Part II* don't rank each other highly — they're structurally different stories (a contained succession drama vs. a dual-timeline origin story), and plot-text alone has no way to know they're the same saga.
- **Every automated OMDb match still needs validation.** Even API responses OMDb itself labeled as "exact" matches were sometimes wrong — a query for "Summer" silently returned "One Crazy Summer." A three-check validation layer (title similarity, year drift, non-movie genre flags) caught 14 bad matches out of 1,870 fetched rows before they could pollute the recommender.

## 🗺️ Table of Contents

- [📜 Project Overview](#-project-overview)
- [✨ Key Findings](#-key-findings)
- [🚀 Getting Started](#-getting-started)
- [📦 The Data](#-the-data)
- [🏗️ Pipeline Architecture](#️-pipeline-architecture)
- [🧹 Step 0 — Enrichment & Validation](#-step-0--enrichment--validation)
- [🧠 Step 1 — Embeddings](#-step-1--embeddings)
- [🔍 Step 2 — Search & Recommend](#-step-2--search--recommend)
- [🪜 The Fallback Ladder](#-the-fallback-ladder)
- [⚠️ Known Limitations](#️-known-limitations)
- [🔮 Planned Enhancements](#-planned-enhancements)
- [📁 File Structure](#-file-structure)
- [📋 Conclusion](#-conclusion)
- [📄 License](#-license)

## 🚀 Getting Started

### Prerequisites

- **Python 3.x**
- **A free [OMDb API key](https://www.omdbapi.com/apikey.aspx)**
- **Required libraries:**

```
pip install pandas numpy requests python-dotenv rapidfuzz sentence-transformers scikit-learn
```

### Setup

1. **Clone the repository:**
```
git clone <your-repo-url>
cd <repo-folder>
```

2. **Create a `.env` file** in the project root (never committed — see `.gitignore`):
```
OMDB_API_KEY=your_key_here
```

3. **Run the pipeline in order** — each step depends on the previous step's output:
```
python enrich_omdb.py       # Step 0a: fetch plots/cast/genre from OMDb (resumable, ~950 requests/day)
python clean_data.py        # Step 0b: validate matches, produce movie_index.csv
python build_embeddings.py  # Step 1: embed all plots, cache movie_embeddings.npy
jupyter notebook Step_2.ipynb  # Step 2: search, recommend, fallback ladder
```

`enrich_omdb.py` is rate-limited by OMDb's free tier and checkpoints its own progress — running it once per day until it finishes is expected, not a bug.

## 📦 The Data

| | |
|---|---|
| **Base source** | IMDb dataset, 1980–2020, 7,668 films |
| **Enrichment source** | OMDb API — full plot text, complete cast, multi-genre tags, writer |
| **Original fields** | name, rating (MPAA), genre (single), year, score, votes, director, writer, star (single), budget, gross, company, runtime |
| **Enriched fields added** | Plot (full-length), Actors (full cast), Genre (multi-label), Writer, Director, imdbRating |

**Why enrichment was necessary, not optional:** the original CSV has exactly one genre/director/writer/star per film and zero plot text. That's enough for a metadata recommender, but not enough to answer "does this feel like that movie" — which is the actual question this project set out to answer.

## 🏗️ Pipeline Architecture

```
IMDB__1980-2020_.csv
        │
        ▼
enrich_omdb.py ──────────► imdb_enriched.csv (+ omdb_unmatched.csv)
   (OMDb fetch,                    │
    resumable,                     ▼
    type=movie,              clean_data.py ──────► movie_index.csv (+ dropped_rows.csv)
    plot=full)                (3-check validation)         │
                                                             ▼
                                                    build_embeddings.py
                                                             │
                                                             ▼
                                                    movie_embeddings.npy
                                                             │
                                                             ▼
                                              Step_2.ipynb — search & recommend
                                        (local match → OMDb → Wikipedia → user paste)
```

## 🧹 Step 0 — Enrichment & Validation

`enrich_omdb.py` fetches per-film data from OMDb with three deliberate safeguards, each added in response to a real failure found during development:

- **`type=movie`** — without it, OMDb's title lookup can silently return a TV episode or short film sharing the queried title instead of the actual movie (caught 6 cases: a talk-show episode matched to *Mad Max 2*, among others).
- **`plot=full`** — the default short plot is sometimes too thin in vocabulary to support meaningful similarity comparison (see [Key Findings](#-key-findings)).
- **Resumable checkpointing** — OMDb's free tier caps daily requests; the script tracks progress in `omdb_progress.json` and stops cleanly rather than erroring out, resuming automatically on the next run.

`clean_data.py` then runs every fetched row through three independent validation checks — any one tripping drops the row into `dropped_rows.csv` with a documented reason instead of silently entering the dataset:

| Check | Catches |
|---|---|
| Title similarity (fuzzy match, floor: 55) | OMDb's own loose internal matching returning the wrong film (e.g. "Summer" → "One Crazy Summer") |
| Year drift (ceiling: ±1 year) | Wrong-entry matches with mismatched release years |
| Non-movie genre markers | TV episodes / shorts / talk-shows that slipped past `type=movie` |

**Result on the tested batch:** 14 of 1,870 fetched rows (0.7%) failed validation and were excluded — each with a specific, reviewable reason, not a silent drop.

## 🧠 Step 1 — Embeddings

Plot text is embedded using `sentence-transformers` (`all-MiniLM-L6-v2`) — chosen for being fast and CPU-friendly at this dataset's scale, producing 384-dimensional vectors. `build_embeddings.py` embeds the full clean dataset once; a hard assertion checks the embedding matrix and `movie_index.csv` always have matching row counts, since any drift between them silently breaks every downstream recommendation (this happened once during development from a file-locking issue — see [Known Limitations](#️-known-limitations)).

## 🔍 Step 2 — Search & Recommend

Given a movie already in the dataset, cosine similarity ranks every other film's embedding against it — a single matrix operation, effectively instant even at full dataset scale. Fuzzy title matching (with lowercase + leading-article normalization) handles minor typos and casing, tuned specifically to reject false positives like `"The Godfather"` incorrectly matching `"The Good Mother"` (a real case found during testing — a naive fuzzy threshold scored that false match at 78.6, indistinguishable from genuine typo-tolerant matches, until normalization separated them cleanly).

## 🪜 The Fallback Ladder

If a queried title isn't in the local dataset, the system doesn't just fail — it escalates through progressively more effortful sources, stopping at the first one that succeeds:

1. **Local fuzzy match** — instant, no API call
2. **Live OMDb lookup** — same validated fetch logic as the bulk pipeline, single call
3. **Wikipedia summary** — fallback if OMDb has no entry
4. **User-pasted synopsis** — if the person has a synopsis from elsewhere (Wikipedia, IMDb's own page) to paste in
5. **User-described plot** — last resort, freeform description

Any movie found via tiers 2–5 is embedded live and **permanently added to the dataset** (`live_additions.csv`, merged back in on every `clean_data.py` rebuild) — so the next search for it hits the fast local path instead of repeating the lookup.

## ⚠️ Known Limitations

Documented deliberately, not hidden:

- **Plot-only similarity misses franchise/sequel relationships.** *Godfather Part I* and *Part II* don't rank each other highly despite being the same saga — see [Planned Enhancements](#-planned-enhancements).
- **Full-length plots can dilute thematic signal** for films whose short plot was already a concentrated genre signature (the Shining regression case). This is an inherent trade-off of mean-pooled sentence embeddings on longer, multi-topic text, not yet resolved.
- **Enrichment coverage is capped by the original dataset's 1980–2020 range.** Pre-1980 or post-2020 films (e.g. a live search for *The Godfather*, 1972) can only ever be added via the live fallback ladder, never the bulk fetch.
- **Bulk enrichment is a multi-day process** (OMDb's free tier rate limit), so dataset completeness — and therefore recommendation quality — improves incrementally over the course of the fetch, not all at once.

## 🔮 Planned Enhancements

- **Hybrid scoring** — a capped metadata bonus (shared director +0.05, shared lead actor +0.03, shared genre up to +0.04) added on top of plot similarity, designed specifically to resolve the Godfather I/II gap without reintroducing the original metadata-only failure mode. *Designed, not yet implemented.*
- **Chunk-level plot matching** — embedding plot sentences individually and scoring by best-matching pair, rather than one averaged whole-plot vector, to directly address the dilution problem found with full-length plots.
- **Two-stage re-ranking** — cosine similarity for fast candidate retrieval, then a cross-encoder model for more accurate re-ranking of just the top 20 candidates.
- **LLM-generated theme tags** — a lightweight proxy for the "keywords" field this dataset never had, to catch conceptual matches (e.g. "unreliable narrator," "heist") that plot vocabulary alone misses.

## 📁 File Structure

| File | Purpose |
|---|---|
| `IMDB__1980-2020_.csv` | Original raw dataset |
| `enrich_omdb.py` | Bulk OMDb enrichment (resumable) |
| `imdb_enriched.csv` / `omdb_unmatched.csv` | Raw enrichment output, pre-validation |
| `clean_data.py` | Validation + cleaning (Step 0) |
| `movie_index.csv` / `dropped_rows.csv` | Validated dataset + audit trail of exclusions |
| `build_embeddings.py` | Plot embedding (Step 1) |
| `movie_embeddings.npy` | Cached embedding matrix |
| `live_additions.csv` | Movies added via the fallback ladder, persisted across rebuilds |
| `Step_2.ipynb` | Search, recommend, fallback ladder |

## 📋 Conclusion

The working system today is a plot-embedding recommender with a validated data pipeline and a documented fallback path for anything outside the dataset. It's deliberately not presented as a finished, polished product — it's a record of testing an assumption (metadata is a good similarity proxy), finding it wrong, testing the fix (plot embeddings), finding *that* has its own failure modes (thin vocabulary, dilution, missed sequels), and designing — but not yet shipping — the next fix for each one. The unfinished pieces in [Planned Enhancements](#-planned-enhancements) are the actual next steps, not aspirational filler.

## 📄 License

This project is licensed under the MIT License.
