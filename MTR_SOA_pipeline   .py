# =============================================================================
# TWO-STAGE EXTRACTIVE SUMMARIZATION  (REVISED PIPELINE)
# Modified TextRank (MTR) + Snake-Optimizer-inspired subset search (SOA)
# =============================================================================
# MODULE 0 — INSTALLATION & IMPORTS
# =============================================================================
!pip install rouge-score bert-score sentence-transformers nltk networkx scikit-learn prettytable tqdm pandas numpy scipy -q

import os
import re
import sys
import json
import math
import time
import zlib
import platform
import warnings
from copy import deepcopy
from collections import defaultdict
from time import perf_counter as pc

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from scipy import stats as sp_stats
from prettytable import PrettyTable

import nltk
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from rouge_score import rouge_scorer as rouge_lib

warnings.filterwarnings("ignore")

for _pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass


# =============================================================================
# MODULE 1 — CONFIGURATION & HYPERPARAMETERS
# =============================================================================

CFG = {
    # ── Data ─────────────────────────────────────────────────────────────────
    # One entry per dataset. Datasets whose file is missing are skipped.
    "datasets": [
        {"name": "BBCNEWS",      "path": "/content/bbc_news_data.csv",
         "text_col": "text",    "summary_col": "summary"},
        {"name": "CNNDAILYMAIL", "path": "/content/cnndm_test.csv",
         "text_col": "text", "summary_col": "summary"},
    ],
    "datasets_to_run"   : None,      # None = all; or e.g. ["BBCNEWS"]
    "encoding"          : "utf-8",   # falls back to latin1 automatically
    "max_samples"       : 100,      # None = all rows; int = subset
    "subsample_mode"    : "random",  # "random" (seeded) or "head"
    "master_seed"       : 42,        # data subsampling and bootstrap

    # ── Randomness ───────────────────────────────────────────────────────────
    "seeds"             : list(range(10)),   # one full run per seed

    # ── MTR: personalization blending weights (must sum to 1) ────────────────
    "alpha_pos"         : 0.30,
    "alpha_lex"         : 0.30,
    "alpha_sem"         : 0.40,

    # ── MTR: PageRank graph ──────────────────────────────────────────────────
    "pagerank_alpha"    : 0.85,
    "sim_threshold"     : 0.10,
    "pagerank_max_iter" : 200,

    # ── Summary length ───────────────────────────────────────────────────────
    "summary_ratio"     : 0.25,          # rho
    "lead_n"            : 3,             # n for Lead-n baseline

    # ── SOA settings ─────────────────────────────────────────────────────────
    "num_snakes"        : 20,
    "max_iter"          : 100,           # T (fixed budget)
    "exploration_rate"  : 0.40,          # eta
    "update_prob"       : 0.50,          # per-slot update probability
    "soa_patience"      : None,          # None = always run T iterations;
                                         # int = stop after that many iterations
                                         # without improvement of the best fitness

    # ── SOA intrinsic fitness weights (must sum to 1) ────────────────────────
    "beta_rank"         : 0.50,
    "beta_pos"          : 0.30,
    "beta_div"          : 0.20,
    # False = R(S) is the raw mean MTR score (scale about 1/N), as submitted.
    # True  = R(S) is divided by max MTR score, so it lives on [0, 1] like P and D.
    "fitness_rescale"   : False,

    # ── SBERT ────────────────────────────────────────────────────────────────
    "sbert_model"       : "all-MiniLM-L6-v2",
    "sbert_device"      : None,          # None = automatic

    # ── Evaluation ───────────────────────────────────────────────────────────
    "run_bertscore"     : False,         # GPU recommended
    "bertscore_rescale" : True,          # rescale with baseline (more readable)
    "bertscore_lang"    : "en",

    # ── Which experiments to run ─────────────────────────────────────────────
    "run_oracle"        : True,
    "run_ablation"      : True,
    "run_length_sensitivity"      : True,
    "length_ratios"               : [0.10, 0.15, 0.35],   # main ratio comes from main run
    "length_seeds"                : 3,
    "run_ref_length_diagnostic"   : True,

    # ── Statistics ───────────────────────────────────────────────────────────
    "bootstrap_n"       : 1000,
    "bootstrap_ci"      : 0.95,
    "alpha_test"        : 0.05,

    # ── Positional analysis ──────────────────────────────────────────────────
    "pos_conc_threshold": 0.30,          # first 30% of the document counts as "early"
    "match_f1_threshold": 0.50,          # min unigram F1 to call a reference sentence
                                         # "matched" to a source sentence

    # ── Output ───────────────────────────────────────────────────────────────
    "output_dir"        : "results",
}

PROPOSED = "MTR+SOA (Proposed)"
REFERENCE_SYSTEM = PROPOSED

assert abs(CFG["alpha_pos"] + CFG["alpha_lex"] + CFG["alpha_sem"] - 1.0) < 1e-6
assert abs(CFG["beta_rank"] + CFG["beta_pos"] + CFG["beta_div"] - 1.0) < 1e-6
assert 0 < CFG["summary_ratio"] < 1


def _renorm(*w):
    s = float(sum(w))
    return tuple(x / s for x in w)


# =============================================================================
# MODULE 2 — DATA LOADING, PREPROCESSING AND PER-DOCUMENT CACHE
# =============================================================================
# Each document is preprocessed ONCE. The component scores and the similarity
# graph do not depend on the random seed or on the ablation variant, so they are
# cached on the document object. Stage times are recorded for the cost analysis.

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
_WN_MAP = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}


def batch_get_wordnet_pos(tokens):
    """Batch POS tagging; falls back to NOUN on any tagger failure.
    Tagging is applied to lowercased, stop-word-free tokens, so the context
    available to the tagger is limited. State this in the manuscript."""
    try:
        tagged = pos_tag(tokens)
    except Exception:
        return [wordnet.NOUN] * len(tokens)
    return [_WN_MAP.get(t[1][0].upper(), wordnet.NOUN) for t in tagged]


def preprocess_sentences(sentences):
    """Lowercase, strip punctuation, remove stop words, POS-aware lemmatize.
    Bigrams are formed AFTER stop-word removal (they are not adjacent word pairs
    of the original sentence)."""
    clean_words, bigrams = [], []
    for sent in sentences:
        cleaned = re.sub(r"[^\w\s]", "", sent.lower())
        raw = [w for w in word_tokenize(cleaned) if w not in STOP_WORDS and w.isalpha()]
        wn_pos = batch_get_wordnet_pos(raw)
        toks = [LEMMATIZER.lemmatize(w, p) for w, p in zip(raw, wn_pos)]
        clean_words.append(toks)
        bigrams.append(set(zip(toks, toks[1:])) if len(toks) > 1 else set())
    return clean_words, bigrams


def _clean_sentence(s):
    return " ".join(s.split())


def _minmax_norm(v):
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn + 1e-9)


def position_scores(n):
    """pos(i) = 1 / log2(i + 2)."""
    return np.array([1.0 / np.log2(i + 2) for i in range(n)])


def bigram_overlap_scores(bigrams):
    """lex(i) = mean over j != i of Jaccard(B_i, B_j); empty sets score 0 and
    are excluded from other sentences' means."""
    n = len(bigrams)
    scores = np.zeros(n)
    for i in range(n):
        if not bigrams[i]:
            continue
        ov = []
        for j in range(n):
            if i == j or not bigrams[j]:
                continue
            union = len(bigrams[i] | bigrams[j])
            ov.append(len(bigrams[i] & bigrams[j]) / union if union else 0.0)
        scores[i] = np.mean(ov) if ov else 0.0
    return scores


def sbert_centrality_scores(sbert, sentences):
    """sem(i) = mean cosine similarity of sentence i to all other sentences."""
    emb = sbert.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    sim = cos_sim(emb).astype(float)
    np.fill_diagonal(sim, 0.0)
    n = len(sentences)
    return sim.sum(axis=1) / max(n - 1, 1)


def build_similarity_matrix(clean_words, threshold):
    """TF-IDF cosine similarity; entries below threshold and the diagonal are zeroed."""
    corpus = [" ".join(w) for w in clean_words]
    n = len(corpus)
    if len(set(corpus)) < 2:
        return np.zeros((n, n))
    try:
        vec = TfidfVectorizer().fit_transform(corpus)
    except ValueError:                       # empty vocabulary
        return np.zeros((n, n))
    sim = cos_sim(vec).astype(float)
    sim[sim < threshold] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim


class Doc:
    __slots__ = ("idx", "text", "reference", "sentences", "clean_words", "bigrams",
                 "n", "ref_sents", "ref_flat", "ref_nl", "ref_words",
                 "pos_s", "lex_s", "sem_s", "sim",
                 "t_seg", "t_pre", "t_lexpos", "t_sem", "t_graph",
                 "mtr_cache", "eval_cache")


def build_doc(i, text, reference, cfg, sbert):
    d = Doc()
    d.idx, d.text, d.reference = i, text, reference
    t0 = pc()
    d.sentences = [_clean_sentence(s) for s in sent_tokenize(text)]
    d.sentences = [s for s in d.sentences if s]
    d.t_seg = pc() - t0
    d.n = len(d.sentences)
    d.ref_sents = [_clean_sentence(s) for s in sent_tokenize(reference)]
    d.ref_sents = [s for s in d.ref_sents if s]
    if d.n == 0 or not d.ref_sents:
        return None
    d.ref_flat = " ".join(d.ref_sents)
    d.ref_nl = "\n".join(d.ref_sents)
    d.ref_words = len(d.ref_flat.split())

    t0 = pc()
    d.clean_words, d.bigrams = preprocess_sentences(d.sentences)
    d.t_pre = d.t_seg + (pc() - t0)

    t0 = pc()
    d.pos_s = _minmax_norm(position_scores(d.n))
    d.lex_s = _minmax_norm(bigram_overlap_scores(d.bigrams))
    d.t_lexpos = pc() - t0

    t0 = pc()
    d.sem_s = _minmax_norm(sbert_centrality_scores(sbert, d.sentences))
    d.t_sem = pc() - t0

    t0 = pc()
    d.sim = build_similarity_matrix(d.clean_words, cfg["sim_threshold"])
    d.t_graph = pc() - t0

    d.mtr_cache, d.eval_cache = {}, {}
    return d


def load_dataset(ds, cfg):
    try:
        df = pd.read_csv(ds["path"], encoding=cfg["encoding"])
    except UnicodeDecodeError:
        df = pd.read_csv(ds["path"], encoding="latin1")
    tc, sc = ds["text_col"], ds["summary_col"]
    df = df[[tc, sc]].dropna()
    df = df[(df[tc].astype(str).str.strip() != "") & (df[sc].astype(str).str.strip() != "")]
    ms = ds.get("max_samples", cfg["max_samples"])
    if ms and len(df) > ms:
        if cfg["subsample_mode"] == "random":
            df = df.sample(n=ms, random_state=cfg["master_seed"]).sort_index()
        else:
            df = df.head(ms)
    return df[tc].astype(str).tolist(), df[sc].astype(str).tolist()


def load_sbert(cfg):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["sbert_model"], device=cfg["sbert_device"])
    model.encode(["warm-up sentence."], show_progress_bar=False)   # exclude load cost from timing
    return model


def build_docs(texts, refs, cfg, sbert):
    docs = []
    for i, (t, r) in enumerate(tqdm(list(zip(texts, refs)), desc="  preprocessing")):
        d = build_doc(i, t, r, cfg, sbert)
        if d is not None:
            docs.append(d)
    return docs


def k_for(n, ratio):
    """k = ceil(rho * N), as stated in the manuscript (clipped to [1, N])."""
    return int(min(n, max(1, math.ceil(ratio * n - 1e-12))))


# =============================================================================
# MODULE 3 — MODIFIED TEXTRANK (PERSONALIZED PAGERANK)
# =============================================================================
# The sentence graph is UNDIRECTED and weighted (symmetric TF-IDF cosine matrix).
# The manuscript currently says "directed"; correct that sentence.

def get_mtr_scores(doc, spec, cfg):
    """Returns (scores, seconds, fell_back). Cached per document and per
    (alphas, uniform, self_loops) so each configuration is computed once."""
    key = (spec["alphas"], spec["uniform"], spec["self_loops"])
    hit = doc.mtr_cache.get(key)
    if hit is not None:
        return hit
    t0 = pc()
    n = doc.n
    a_pos, a_lex, a_sem = spec["alphas"]
    if spec["uniform"]:
        pers = np.ones(n) / n
        pers_dict = None
    else:
        pers = a_pos * doc.pos_s + a_lex * doc.lex_s + a_sem * doc.sem_s
        tot = pers.sum()
        pers = pers / tot if tot > 0 else np.ones(n) / n
        pers_dict = {i: float(pers[i]) for i in range(n)}
    G = nx.from_numpy_array(doc.sim)
    if spec["self_loops"]:
        for i in range(n):
            G.add_edge(i, i, weight=1e-6)
    fell_back = False
    try:
        pr = nx.pagerank(G, alpha=cfg["pagerank_alpha"], personalization=pers_dict,
                         max_iter=cfg["pagerank_max_iter"])
        scores = np.array([pr[i] for i in range(n)])
    except nx.PowerIterationFailedConvergence:
        scores, fell_back = pers.copy(), True
    scores = scores / (scores.sum() + 1e-9)
    out = (scores, pc() - t0, fell_back)
    doc.mtr_cache[key] = out
    return out


# =============================================================================
# MODULE 4 — SUBSET SELECTION: INTRINSIC FITNESS, SOA-STYLE SEARCH, RANDOM SEARCH
# =============================================================================

def _fit_vector(scores, cfg):
    """R(S) uses the raw MTR scores (sum = 1) unless fitness_rescale is True."""
    return scores / (scores.max() + 1e-12) if cfg["fitness_rescale"] else scores


def fitness_batch(sel, r_fit, pos_reward, n, cfg):
    """F(S) = -(b_rank*R + b_pos*P + b_div*D) for every row of `sel` (E x k).
    R = mean r(i);  P = mean 1/(i+1);  D = std(indices)/N.  Lower is better."""
    rank = r_fit[sel].mean(axis=1)
    pos = pos_reward[sel].mean(axis=1)
    div = sel.std(axis=1) / (n + 1e-9)
    return -(cfg["beta_rank"] * rank + cfg["beta_pos"] * pos + cfg["beta_div"] * div)


def fitness_components(sel, scores, n, cfg):
    """Weighted contribution of each reward term for one selected subset."""
    sel = np.asarray(sel, dtype=int)
    r_fit = _fit_vector(scores, cfg)
    rank = float(r_fit[sel].mean())
    pos = float((1.0 / (sel + 1.0)).mean())
    div = float(sel.std() / (n + 1e-9))
    return cfg["beta_rank"] * rank, cfg["beta_pos"] * pos, cfg["beta_div"] * div


def sel_soa(doc, k, rng, cfg, scores):
    """Population search with improvement-only updates (see header note).

    Init     : P snakes, each k distinct indices drawn uniformly.
    Update   : each slot is replaced with prob `update_prob`; the replacement is
               uniform with prob eta, else drawn proportional to the MTR scores.
    Repair   : duplicates are replaced by random unused indices.
    Accept   : a snake moves only if its fitness strictly decreases.
    Stopping : T iterations, or `soa_patience` iterations without improvement of
               the global best (if patience is not None).
    """
    n = doc.n
    if k >= n:
        return np.arange(n), {"last_improve": 0, "iters": 0}
    P, T = cfg["num_snakes"], cfg["max_iter"]
    eta, pu, patience = cfg["exploration_rate"], cfg["update_prob"], cfg["soa_patience"]
    s = scores.sum()
    probs = scores / s if s > 0 else np.full(n, 1.0 / n)
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0
    r_fit = _fit_vector(scores, cfg)
    pos_reward = 1.0 / (np.arange(n) + 1.0)

    pop = np.argpartition(rng.random((P, n)), k - 1, axis=1)[:, :k]
    fit = fitness_batch(pop, r_fit, pos_reward, n, cfg)
    b = int(np.argmin(fit))
    best_pos, best_fit, last, stall, iters = pop[b].copy(), float(fit[b]), 0, 0, 0

    for t in range(1, T + 1):
        iters = t
        mask = rng.random((P, k)) < pu
        explore = rng.random((P, k)) < eta
        rand_idx = rng.integers(0, n, size=(P, k))
        guided = np.minimum(np.searchsorted(cdf, rng.random((P, k))), n - 1)
        cand = np.where(mask, np.where(explore, rand_idx, guided), pop)

        srt = np.sort(cand, axis=1)
        dup = (srt[:, 1:] == srt[:, :-1]).any(axis=1)
        if dup.any():
            D = np.flatnonzero(dup)
            M = rng.random((len(D), n))
            M[np.arange(len(D))[:, None], cand[D]] = 2.0    # keep present indices
            cand[D] = np.argpartition(-M, k - 1, axis=1)[:, :k]

        cfit = fitness_batch(cand, r_fit, pos_reward, n, cfg)
        better = cfit < fit
        if better.any():
            pop[better] = cand[better]
            fit[better] = cfit[better]
        j = int(np.argmin(fit))
        if fit[j] < best_fit:
            best_fit, best_pos, last, stall = float(fit[j]), pop[j].copy(), t, 0
        else:
            stall += 1
        if patience is not None and stall >= patience:
            break
    return best_pos, {"last_improve": last, "iters": iters}


def sel_random_search(doc, k, rng, cfg, scores):
    """Control: same fitness and the same number of evaluations as sel_soa."""
    n = doc.n
    if k >= n:
        return np.arange(n), {"last_improve": 0, "iters": 0}
    E = cfg["num_snakes"] * (cfg["max_iter"] + 1)
    cand = np.argpartition(rng.random((E, n)), k - 1, axis=1)[:, :k]
    fit = fitness_batch(cand, _fit_vector(scores, cfg), 1.0 / (np.arange(n) + 1.0), n, cfg)
    return cand[int(np.argmin(fit))], {"last_improve": int(np.argmin(fit)), "iters": E}


def sel_greedy(doc, k, rng, cfg, scores):
    return np.argsort(-scores, kind="stable")[:k], {}


def sel_lead_n(doc, k, rng, cfg, scores):
    return np.arange(min(cfg["lead_n"], doc.n)), {}


def sel_lead_k(doc, k, rng, cfg, scores):
    return np.arange(min(k, doc.n)), {}


def sel_random_k(doc, k, rng, cfg, scores):
    return rng.choice(doc.n, size=min(k, doc.n), replace=False), {}


def sel_lexrank(doc, k, rng, cfg, scores):
    """LexRank approximation: thresholded binary adjacency + degree centrality.
    (Not the full IDF-modified-cosine LexRank; label it as an approximation.)"""
    adj = (doc.sim > cfg["sim_threshold"]).astype(float)
    deg = adj.sum(axis=1) / max(doc.n - 1, 1)
    return np.argsort(-deg, kind="stable")[:k], {}


_ORACLE_SCORER = rouge_lib.RougeScorer(["rouge1"], use_stemmer=True)


def sel_oracle(doc, k, rng, cfg, scores):
    """Greedy ROUGE-1-F oracle. Sees the reference at selection time; it is an
    upper-bound diagnostic and is never used as a fitness signal."""
    remaining, selected = list(range(doc.n)), []
    for _ in range(min(k, doc.n)):
        best_s, best_i = -1.0, remaining[0]
        for i in remaining:
            cand = " ".join(doc.sentences[j] for j in sorted(selected + [i]))
            f = _ORACLE_SCORER.score(doc.ref_flat, cand)["rouge1"].fmeasure
            if f > best_s:
                best_s, best_i = f, i
        selected.append(best_i)
        remaining.remove(best_i)
    return np.array(selected), {}


# =============================================================================
# MODULE 5 — SYSTEM DEFINITIONS (BASELINES, PROPOSED METHOD, ABLATIONS)
# =============================================================================

def _mtr_needs(alphas, uniform):
    if uniform:
        return ["pre", "graph"]
    needs = ["pre", "graph"]
    if alphas[0] > 0 or alphas[1] > 0:
        needs.append("lexpos")
    if alphas[2] > 0:
        needs.append("sem")
    return needs


def make_mtr_system(fn, alphas, uniform=False, self_loops=True, over=None,
                    stochastic=True):
    return {"fn": fn, "stochastic": stochastic, "over": over or {},
            "mtr": {"alphas": tuple(round(a, 10) for a in alphas),
                    "uniform": uniform, "self_loops": self_loops},
            "needs": _mtr_needs(alphas, uniform)}


def build_baselines(cfg):
    S = {
        "Lead-3": {"fn": sel_lead_n, "stochastic": False, "over": {}, "mtr": None, "needs": ["seg"]},
        "Lead-k (matched length)": {"fn": sel_lead_k, "stochastic": False, "over": {}, "mtr": None, "needs": ["seg"]},
        "Random-k": {"fn": sel_random_k, "stochastic": True, "over": {}, "mtr": None, "needs": ["seg"]},
        "Standard TextRank": make_mtr_system(sel_greedy, (0, 0, 0), uniform=True,
                                             self_loops=False, stochastic=False),
        "LexRank (approx.)": {"fn": sel_lexrank, "stochastic": False, "over": {}, "mtr": None,
                              "needs": ["pre", "graph"]},
    }
    if cfg["run_oracle"]:
        S["Oracle (in-house)"] = {"fn": sel_oracle, "stochastic": False, "over": {},
                                  "mtr": None, "needs": ["seg"]}
    return S


def build_proposed(cfg):
    base = (cfg["alpha_pos"], cfg["alpha_lex"], cfg["alpha_sem"])
    return {PROPOSED: make_mtr_system(sel_soa, base)}


def build_ablations(cfg):
    base = (cfg["alpha_pos"], cfg["alpha_lex"], cfg["alpha_sem"])

    def drop(i):                                   # proportional redistribution
        v = list(base)
        v[i] = 0.0
        return _renorm(*v)

    b = (cfg["beta_rank"], cfg["beta_pos"], cfg["beta_div"])

    def bset(r, p, d):
        return {"beta_rank": r, "beta_pos": p, "beta_div": d}

    A = {
        "A1_no_position":   make_mtr_system(sel_soa, drop(0)),
        "A2_no_bigram":     make_mtr_system(sel_soa, drop(1)),
        "A3_no_sbert":      make_mtr_system(sel_soa, drop(2)),
        "A4_no_soa":        make_mtr_system(sel_greedy, base, stochastic=False),
        "A5_random_search": make_mtr_system(sel_random_search, base),
        "A6_uniform_prior_soa": make_mtr_system(sel_soa, (0, 0, 0), uniform=True),
        "A7_fitness_no_position": make_mtr_system(sel_soa, base, over=bset(*_renorm(b[0], 0, b[2]))),
        "A8_fitness_no_diversity": make_mtr_system(sel_soa, base, over=bset(*_renorm(b[0], b[1], 0))),
        "A9_fitness_rank_only": make_mtr_system(sel_soa, base, over=bset(1.0, 0.0, 0.0)),
        "A10_fitness_rescaled_rank": make_mtr_system(sel_soa, base, over={"fitness_rescale": True}),
        "A11_rescaled_no_position": make_mtr_system(
            sel_soa, base, over={"fitness_rescale": True, **bset(*_renorm(b[0], 0, b[2]))}),
    }
    return A


# =============================================================================
# MODULE 6 — EVALUATION (ROUGE, optional BERTScore)
# =============================================================================
# Scores are stored on a 0-1 scale and reported on 0-100 everywhere.
# ROUGE-1/2/L : whole summary as one token sequence.
# ROUGE-Lsum  : summary-level LCS, one sentence per line (rouge-score convention).

METRIC_TYPES = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
METRIC_KEYS = [f"{t}_{s}" for t in METRIC_TYPES for s in ("p", "r", "f")]
MIDX = {k: i for i, k in enumerate(METRIC_KEYS)}
_SC_FLAT = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_SC_SUM = rouge_lib.RougeScorer(["rougeLsum"], use_stemmer=True)


def evaluate_selection(doc, sel):
    key = tuple(int(i) for i in sel)
    hit = doc.eval_cache.get(key)
    if hit is not None:
        return hit
    hyp_flat = " ".join(doc.sentences[i] for i in key)
    hyp_nl = "\n".join(doc.sentences[i] for i in key)
    a = _SC_FLAT.score(doc.ref_flat, hyp_flat)
    b = _SC_SUM.score(doc.ref_nl, hyp_nl)
    vec = []
    for t in ["rouge1", "rouge2", "rougeL"]:
        vec += [a[t].precision, a[t].recall, a[t].fmeasure]
    vec += [b["rougeLsum"].precision, b["rougeLsum"].recall, b["rougeLsum"].fmeasure]
    out = (np.array(vec), len(hyp_flat.split()))
    doc.eval_cache[key] = out
    return out


def compute_bertscore(res, docs, cfg):
    """BERTScore-F1 for the first run of a system (all systems use the same rule)."""
    from bert_score import score as bs_fn
    hyps, refs = [], []
    for d, doc in enumerate(docs):
        hyps.append(" ".join(doc.sentences[i] for i in res["sel_run0"][d]))
        refs.append(doc.ref_flat)
    _, _, F1 = bs_fn(hyps, refs, lang=cfg["bertscore_lang"], verbose=False,
                     rescale_with_baseline=cfg["bertscore_rescale"])
    return float(F1.mean()) * 100.0


# =============================================================================
# MODULE 7 — SYSTEM RUNNER
# =============================================================================

TIME_KEY = {"seg": "t_seg", "pre": "t_pre", "lexpos": "t_lexpos",
            "sem": "t_sem", "graph": "t_graph"}


def run_system(name, spec, docs, cfg, seeds, k_fn, keep_sel=False):
    n_runs = len(seeds) if spec["stochastic"] else 1
    nd = len(docs)
    run_cfg = {**cfg, **spec["over"]}
    metrics = np.zeros((n_runs, nd, len(METRIC_KEYS)))
    words = np.zeros((n_runs, nd))
    times = np.zeros((n_runs, nd))
    prof_pos = np.zeros((n_runs, nd))
    prof_early = np.zeros((n_runs, nd))
    ks = np.zeros((n_runs, nd))
    comps = np.full((n_runs, nd, 3), np.nan)
    last_imp = np.full((n_runs, nd), np.nan)
    iters_used = np.full((n_runs, nd), np.nan)
    fallbacks, sel_run0 = 0, []

    for r in range(n_runs):
        rng = np.random.default_rng(np.random.SeedSequence([int(seeds[r]), zlib.crc32(name.encode())]))
        for d, doc in enumerate(tqdm(docs, desc=f"    {name} [run {r + 1}/{n_runs}]", leave=False)):
            k = k_fn(doc)
            scores, t_mtr = None, 0.0
            if spec["mtr"] is not None:
                scores, t_mtr, fb = get_mtr_scores(doc, spec["mtr"], run_cfg)
                fallbacks += int(fb) if r == 0 else 0
            t0 = pc()
            sel, info = spec["fn"](doc, k, rng, run_cfg, scores)
            t_sel = pc() - t0
            sel = np.sort(np.asarray(sel, dtype=int))
            m, w = evaluate_selection(doc, sel)
            metrics[r, d], words[r, d] = m, w
            times[r, d] = t_sel + t_mtr + sum(getattr(doc, TIME_KEY[x]) for x in spec["needs"])
            ks[r, d] = len(sel)
            prof_pos[r, d] = float(np.mean(sel / (doc.n - 1))) if doc.n > 1 else 0.0
            prof_early[r, d] = float(np.mean(sel < max(1, int(cfg["pos_conc_threshold"] * doc.n))))
            if scores is not None:
                comps[r, d] = fitness_components(sel, scores, doc.n, run_cfg)
            if "last_improve" in info:
                last_imp[r, d], iters_used[r, d] = info["last_improve"], info["iters"]
            if r == 0 and (cfg["run_bertscore"] or keep_sel):
                sel_run0.append(sel)

    res = {"name": name, "stochastic": spec["stochastic"], "metrics": metrics,
           "words": words, "times": times, "prof_pos": prof_pos, "prof_early": prof_early,
           "ks": ks, "comps": comps, "last_improve": last_imp, "iters": iters_used,
           "fallbacks": fallbacks, "sel_run0": sel_run0, "bertscore": None,
           "uses_fitness": spec["fn"] in (sel_soa, sel_random_search), "run_cfg": run_cfg}
    if cfg["run_bertscore"]:
        res["bertscore"] = compute_bertscore(res, docs, cfg)
    return res


# =============================================================================
# MODULE 8 — STATISTICS (CIs, paired tests, Holm correction)
# =============================================================================

def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order, m = np.argsort(p), len(p)
    adj, run = np.empty(m), 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def _pct_bounds(ci):
    a = 1.0 - ci
    return 100 * a / 2, 100 * (1 - a / 2)


def perdoc(res, key):
    """Per-document score averaged over seeds (vector of length n_docs)."""
    return res["metrics"][:, :, MIDX[key]].mean(axis=0)


def summarize(results, cfg, boot_idx):
    lo_q, hi_q = _pct_bounds(cfg["bootstrap_ci"])
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        row = {"Method": name, "n_runs": m.shape[0], "n_docs": m.shape[1]}
        pdm = m.mean(axis=0)
        for key, i in MIDX.items():
            row[key] = 100 * pdm[:, i].mean()
        for t in METRIC_TYPES:
            i = MIDX[f"{t}_f"]
            seed_means = m[:, :, i].mean(axis=1) * 100
            row[f"{t}_f_seed_sd"] = float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
            bm = pdm[:, i][boot_idx].mean(axis=1) * 100
            row[f"{t}_f_ci_lo"], row[f"{t}_f_ci_hi"] = np.percentile(bm, [lo_q, hi_q])
        row["summary_words"] = float(res["words"].mean())
        row["bertscore_f1"] = res["bertscore"] if res["bertscore"] is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def significance_tests(results, cfg, boot_idx, ref_name):
    if ref_name not in results:
        return pd.DataFrame()
    lo_q, hi_q = _pct_bounds(cfg["bootstrap_ci"])
    rows = []
    for t in METRIC_TYPES:
        key = f"{t}_f"
        ref_vec = perdoc(results[ref_name], key)
        block = []
        for name, res in results.items():
            if name == ref_name:
                continue
            other = perdoc(res, key)
            diff = ref_vec - other
            bm = diff[boot_idx].mean(axis=1) * 100
            t_stat, p_t = sp_stats.ttest_rel(ref_vec, other)
            try:
                p_w = float(sp_stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            except ValueError:
                p_w = 1.0
            sd = diff.std(ddof=1)
            block.append({
                "Metric": key, "Comparison": f"{ref_name} vs {name}",
                "Mean_ref": 100 * ref_vec.mean(), "Mean_other": 100 * other.mean(),
                "Mean_diff": 100 * diff.mean(),
                "Diff_CI_lo": np.percentile(bm, lo_q), "Diff_CI_hi": np.percentile(bm, hi_q),
                "t_stat": float(t_stat), "p_ttest": float(p_t), "p_wilcoxon": p_w,
                "Cohen_dz": float(diff.mean() / sd) if sd > 0 else np.nan})
        if block:
            ph_t = holm([b["p_ttest"] for b in block])
            ph_w = holm([b["p_wilcoxon"] for b in block])
            for b, a, c in zip(block, ph_t, ph_w):
                b["p_holm_ttest"], b["p_holm_wilcoxon"] = a, c
                b["significant_holm"] = bool(a < cfg["alpha_test"])
            rows += block
    return pd.DataFrame(rows)


# =============================================================================
# MODULE 9 — DIAGNOSTICS: COST, FITNESS DECOMPOSITION, POSITION PROFILES
# =============================================================================

def cost_table(results, cfg):
    rows = []
    base = results.get("A4_no_soa")
    base_ms = base["times"].mean() * 1000 if base is not None else np.nan
    for name, res in results.items():
        ms = res["times"] * 1000
        li = res["last_improve"]
        row = {"Method": name, "ms_per_doc_mean": float(ms.mean()),
               "ms_per_doc_sd": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
               "ms_per_doc_median": float(np.median(ms)),
               "x_vs_A4_no_soa": float(ms.mean() / base_ms) if base is not None else np.nan,
               "total_seconds_one_run": float(res["times"][0].sum()),
               "pagerank_fallbacks": res["fallbacks"]}
        if not np.all(np.isnan(li)):
            it = res["iters"]
            row.update({"last_improve_iter_mean": float(np.nanmean(li)),
                        "last_improve_iter_median": float(np.nanmedian(li)),
                        "last_improve_iter_p90": float(np.nanpercentile(li, 90)),
                        "share_improving_after_T_half": float(np.nanmean(li > cfg["max_iter"] / 2)),
                        "iterations_used_mean": float(np.nanmean(it))})
        rows.append(row)
    return pd.DataFrame(rows)


def fitness_decomposition_table(results):
    rows = []
    for name, res in results.items():
        c = res["comps"]
        if np.all(np.isnan(c)):
            continue
        mean = np.nanmean(c.reshape(-1, 3), axis=0)
        tot = mean.sum()
        rows.append({"Method": name, "beta_rank*R": mean[0], "beta_pos*P": mean[1],
                     "beta_div*D": mean[2], "total_reward": tot,
                     "share_rank": mean[0] / tot, "share_pos": mean[1] / tot,
                     "share_div": mean[2] / tot})
    return pd.DataFrame(rows)


def selection_profile_table(results):
    rows = []
    for name, res in results.items():
        rows.append({"Method": name,
                     "mean_selected_per_doc": float(res["ks"].mean()),
                     "mean_relative_position": float(res["prof_pos"].mean()),
                     "share_selected_in_early_part": float(res["prof_early"].mean())})
    return pd.DataFrame(rows)


def _f1_sets(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    p, r = inter / len(b), inter / len(a)
    return 2 * p * r / (p + r)


def positional_concentration(docs, cfg):
    """Where do reference sentences come from in the source document?

    exact_share_early   : share of reference sentences that are verbatim source
                          sentences lying in the early part (the submitted metric)
    match_coverage      : share of reference sentences whose best unigram-F1 match
                          (lemmatized, stop words removed) reaches the threshold
    fuzzy_share_early   : among matched reference sentences, share whose best match
                          lies in the early part
    mean_relative_pos   : mean relative position (0 = first, 1 = last) of the matches
    Reading: exact_share_early is near zero for abstractive references by
    construction, so use match_coverage and the fuzzy columns to describe them."""
    thr, f1_min = cfg["pos_conc_threshold"], cfg["match_f1_threshold"]
    exact, tot, matched, early_m, relpos = 0, 0, 0, 0, []
    for doc in docs:
        cutoff = max(1, int(thr * doc.n))
        early_set = set(doc.sentences[:cutoff])
        ref_clean, _ = preprocess_sentences(doc.ref_sents)
        src_sets = [set(w) for w in doc.clean_words]
        for rs, rc in zip(doc.ref_sents, ref_clean):
            tot += 1
            exact += int(rs in early_set)
            rset = set(rc)
            f1s = [_f1_sets(s, rset) for s in src_sets]
            j = int(np.argmax(f1s))
            if f1s[j] >= f1_min:
                matched += 1
                early_m += int(j < cutoff)
                relpos.append(j / (doc.n - 1) if doc.n > 1 else 0.0)
    return {"n_reference_sentences": tot,
            "exact_share_early": exact / tot,
            "match_coverage": matched / tot,
            "fuzzy_share_early": early_m / matched if matched else np.nan,
            "mean_relative_pos_of_matches": float(np.mean(relpos)) if relpos else np.nan,
            "early_threshold": thr, "match_f1_threshold": f1_min}


# =============================================================================
# MODULE 10 — REPORTING
# =============================================================================

def hyperparameter_table(cfg):
    rows = [
        ("Positional weight", "alpha_pos", cfg["alpha_pos"], "MTR personalization"),
        ("Lexical (bigram) weight", "alpha_lex", cfg["alpha_lex"], "MTR personalization"),
        ("Semantic (SBERT) weight", "alpha_sem", cfg["alpha_sem"], "MTR personalization"),
        ("Damping factor", "alpha", cfg["pagerank_alpha"], "PageRank"),
        ("Similarity threshold", "tau", cfg["sim_threshold"], "Graph pruning"),
        ("Max PageRank iterations", "-", cfg["pagerank_max_iter"], "PageRank limit"),
        ("Summary ratio", "rho", cfg["summary_ratio"], "k = ceil(rho * N)"),
        ("Population size", "n_snakes", cfg["num_snakes"], "Search"),
        ("Iterations (fixed budget)", "T", cfg["max_iter"], "Search"),
        ("Exploration rate", "eta", cfg["exploration_rate"], "Search"),
        ("Per-slot update probability", "-", cfg["update_prob"], "Search"),
        ("Patience (early stop)", "-", cfg["soa_patience"], "None = fixed T"),
        ("Rank reward weight", "beta_rank", cfg["beta_rank"], "Fitness"),
        ("Position reward weight", "beta_pos", cfg["beta_pos"], "Fitness"),
        ("Diversity reward weight", "beta_div", cfg["beta_div"], "Fitness"),
        ("Rescale rank reward", "-", cfg["fitness_rescale"], "Fitness"),
        ("Sentence encoder", "-", cfg["sbert_model"], "SBERT (frozen)"),
        ("Random seeds", "-", str(cfg["seeds"]), "Independent full runs"),
        ("Bootstrap resamples", "B", cfg["bootstrap_n"], "CI estimation"),
        ("Early-part fraction", "theta", cfg["pos_conc_threshold"], "Positional analysis"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Symbol", "Value", "Role"])


def collect_environment(cfg):
    import importlib.metadata as md
    libs = ["numpy", "pandas", "scipy", "networkx", "scikit-learn", "nltk",
            "rouge-score", "sentence-transformers", "bert-score", "torch"]
    ver = {}
    for l in libs:
        try:
            ver[l] = md.version(l)
        except Exception:
            ver[l] = "not installed"
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "processor": platform.processor(), "libraries": ver, "gpu": "none"}
    try:
        import torch
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    env["seeds"], env["master_seed"] = cfg["seeds"], cfg["master_seed"]
    return env


def formatted_table(summary):
    rows = []
    for _, r in summary.iterrows():
        row = {"Method": r["Method"]}
        for t, lab in zip(METRIC_TYPES, ["R-1", "R-2", "R-L", "R-Lsum"]):
            f = r[f"{t}_f"]
            sd = r[f"{t}_f_seed_sd"]
            cell = f"{f:.2f}" + (f" ± {sd:.2f}" if r["n_runs"] > 1 else "")
            row[lab] = cell
            row[lab + " 95% CI"] = f"[{r[f'{t}_f_ci_lo']:.2f}, {r[f'{t}_f_ci_hi']:.2f}]"
        row["Words/summary"] = f"{r['summary_words']:.1f}"
        row["BERTScore-F1"] = "-" if np.isnan(r["bertscore_f1"]) else f"{r['bertscore_f1']:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def print_table(df, title, cols, floatfmt=".2f"):
    t = PrettyTable()
    t.title = title
    t.field_names = cols
    for _, r in df.iterrows():
        t.add_row([r[c] if isinstance(r[c], str) else (f"{r[c]:{floatfmt}}" if pd.notna(r[c]) else "-")
                   for c in cols])
    print(f"\n{t}")


def length_sensitivity(results, docs, cfg, systems, ratio_main):
    rows = []

    def add(label, res_dict, n_seeds):
        for name, res in res_dict.items():
            row = {"Setting": label, "Method": name, "n_seeds": n_seeds}
            for t in METRIC_TYPES:
                row[f"{t}_f"] = 100 * perdoc(res, f"{t}_f").mean()
            row["summary_words"] = float(res["words"].mean())
            rows.append(row)

    subset = [PROPOSED, "A4_no_soa", "Standard TextRank", "Lead-k (matched length)"]
    add(f"rho={ratio_main} (main run)", {n: results[n] for n in subset if n in results},
        len(cfg["seeds"]))
    seeds = cfg["seeds"][:cfg["length_seeds"]]
    for ratio in cfg["length_ratios"]:
        out = {n: run_system(n, systems[n], docs, cfg, seeds, lambda d, r=ratio: k_for(d.n, r))
               for n in subset if n in systems}
        add(f"rho={ratio}", out, len(seeds))
    if cfg["run_ref_length_diagnostic"]:
        out = {n: run_system(n, systems[n], docs, cfg, seeds,
                             lambda d: int(min(d.n, max(1, len(d.ref_sents)))))
               for n in subset if n in systems}
        add("k = number of reference sentences (diagnostic)", out, len(seeds))
    return pd.DataFrame(rows)


# =============================================================================
# MODULE 11 — DRIVER
# =============================================================================

def run_dataset(ds, cfg, sbert, out_root):
    name = ds["name"]
    out = os.path.join(out_root, name)
    os.makedirs(out, exist_ok=True)
    print("\n" + "=" * 74 + f"\n  DATASET: {name}\n" + "=" * 74)

    texts, refs = load_dataset(ds, cfg)
    docs = build_docs(texts, refs, cfg, sbert)
    nd = len(docs)
    print(f"  Documents used: {nd}   mean sentences/doc: {np.mean([d.n for d in docs]):.1f}"
          f"   mean reference words: {np.mean([d.ref_words for d in docs]):.1f}")

    hp = hyperparameter_table(cfg)
    hp.to_csv(os.path.join(out, "hyperparameters.csv"), index=False)
    env = collect_environment(cfg)
    with open(os.path.join(out, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)

    systems = {}
    systems.update(build_baselines(cfg))
    systems.update(build_proposed(cfg))
    ablations = build_ablations(cfg) if cfg["run_ablation"] else {}
    systems.update(ablations)

    k_main = lambda d: k_for(d.n, cfg["summary_ratio"])
    results = {}
    print("\n  Running systems ...")
    for sname, spec in systems.items():
        t0 = pc()
        results[sname] = run_system(sname, spec, docs, cfg, cfg["seeds"], k_main)
        r = results[sname]
        print(f"    {sname:<28} R1={100 * perdoc(r, 'rouge1_f').mean():6.2f}  "
              f"R2={100 * perdoc(r, 'rouge2_f').mean():6.2f}  "
              f"RL={100 * perdoc(r, 'rougeL_f').mean():6.2f}  "
              f"words={r['words'].mean():6.1f}  ({pc() - t0:6.1f}s)")

    # statistics
    rng = np.random.default_rng(cfg["master_seed"])
    boot_idx = rng.integers(0, nd, size=(cfg["bootstrap_n"], nd))
    summary = summarize(results, cfg, boot_idx)
    summary["ref_words_mean"] = float(np.mean([d.ref_words for d in docs]))
    sig = significance_tests(results, cfg, boot_idx, REFERENCE_SYSTEM)
    cost = cost_table(results, cfg)
    fdec = fitness_decomposition_table(results)
    prof = selection_profile_table(results)
    pc_res = positional_concentration(docs, cfg)

    summary.to_csv(os.path.join(out, "main_results.csv"), index=False)
    formatted_table(summary).to_csv(os.path.join(out, "table_formatted.csv"), index=False)
    sig.to_csv(os.path.join(out, "significance_tests.csv"), index=False)
    cost.to_csv(os.path.join(out, "computational_cost.csv"), index=False)
    fdec.to_csv(os.path.join(out, "fitness_decomposition.csv"), index=False)
    prof.to_csv(os.path.join(out, "selection_profile.csv"), index=False)
    pd.DataFrame([pc_res]).to_csv(os.path.join(out, "positional_concentration.csv"), index=False)

    wide = {"doc_index": [d.idx for d in docs]}
    for sname, res in results.items():
        for t in METRIC_TYPES:
            wide[f"{sname}__{t}_f"] = perdoc(res, f"{t}_f")
    pd.DataFrame(wide).to_csv(os.path.join(out, "per_document_scores.csv"), index=False)

    if cfg["run_length_sensitivity"]:
        print("\n  Length sensitivity ...")
        ls = length_sensitivity(results, docs, cfg, systems, cfg["summary_ratio"])
        ls.to_csv(os.path.join(out, "length_sensitivity.csv"), index=False)
        print_table(ls, "Length sensitivity (F, 0-100)",
                    ["Setting", "Method", "n_seeds", "rouge1_f", "rouge2_f", "rougeL_f", "summary_words"])

    # console report
    print_table(formatted_table(summary), f"{name}: main results (mean ± SD over seeds; 95% bootstrap CI)",
                ["Method", "R-1", "R-1 95% CI", "R-2", "R-L", "R-Lsum", "Words/summary"], floatfmt="")
    print_table(cost, f"{name}: computational cost",
                ["Method", "ms_per_doc_mean", "x_vs_A4_no_soa", "pagerank_fallbacks"])
    print_table(fdec, f"{name}: fitness decomposition (weighted reward terms)",
                ["Method", "beta_rank*R", "beta_pos*P", "beta_div*D", "share_rank", "share_pos", "share_div"],
                floatfmt=".4f")
    print(f"\n  Positional analysis: {json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in pc_res.items()})}")
    return results


def main():
    out_root = CFG["output_dir"]
    os.makedirs(out_root, exist_ok=True)
    todo = [d for d in CFG["datasets"]
            if CFG["datasets_to_run"] is None or d["name"] in CFG["datasets_to_run"]]
    todo = [d for d in todo if os.path.exists(d["path"])]
    if not todo:
        print("No dataset file found. Edit CFG['datasets'] paths and run again.")
        return {}
    print("Loading SBERT ...")
    sbert = load_sbert(CFG)
    with open(os.path.join(out_root, "environment.json"), "w") as f:
        json.dump(collect_environment(CFG), f, indent=2)
    all_results = {}
    for ds in todo:
        all_results[ds["name"]] = run_dataset(ds, CFG, sbert, out_root)
    print(f"\nDone. Files are in {out_root}/<dataset>/")
    return all_results


if __name__ == "__main__":
    ALL_RESULTS = main()
