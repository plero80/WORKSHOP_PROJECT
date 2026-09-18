from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
from .common import atomic_json, atomic_npz, digest, file_sha, finite, read_json, read_jsonl, response_key


def top_neighbors(query, reference, k):
    """Exact cosine search; ties keep memory-row order."""
    query, reference = np.asarray(query, np.float32), np.asarray(reference, np.float32)
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError('Feature dimensions differ.')
    if k < 1 or k > len(reference):
        raise ValueError('k must fit the nonempty memory.')
    similarities = np.clip(query @ reference.T, -1., 1.)
    indices = np.argsort(-similarities, axis=1, kind='stable')[:, :k]
    return np.take_along_axis(similarities, indices, axis=1), indices


def save_features(path, rows, embeddings, encoder_identity):
    """Persist features by exact scorer/input identity, including the candidate answer."""
    x = np.asarray(embeddings, np.float32)
    if x.ndim != 2 or len(x) != len(rows) or not np.isfinite(x).all():
        raise ValueError('Invalid feature archive.')
    atomic_npz(path, embeddings=x, keys=[response_key(encoder_identity, r) for r in rows],
               encoder_identity=encoder_identity)


def load_features(path, rows, encoder_identity):
    with np.load(path, allow_pickle=False) as data:
        if (data['encoder_identity'].item() != encoder_identity or
            data['keys'].tolist() != [response_key(encoder_identity, r) for r in rows]):
            raise ValueError('Feature archive does not match the encoder/answers.')
        x = data['embeddings'].copy()
    if x.ndim != 2 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4):
        raise ValueError('Invalid normalized features.')
    return x


@dataclass
class Normalization:
    proxy_mean: float
    proxy_std: float
    judge_mean: float
    judge_std: float
    threshold: float

    @classmethod
    def fit(cls, proxy, judge, quantile, minimum_std):
        proxy, judge = np.asarray(proxy, float), np.asarray(judge, float)
        finite(proxy, "calibration proxy ratings")
        finite(judge, "calibration judge ratings")
        if len(proxy) != len(judge) or len(proxy) < 2:
            raise ValueError("Need aligned calibration scores.")
        mp, sp, mj, sj = float(proxy.mean()), float(proxy.std()), float(judge.mean()), float(judge.std())
        if min(sp, sj) < minimum_std:
            raise ValueError(f"Degenerate judge calibration: proxy std={sp:.5f}, judge std={sj:.5f}. This scoring protocol has insufficient variation. Inspect calibration responses; do not fabricate a gap by dividing by an epsilon.")
        gaps = (proxy - mp) / sp - (judge - mj) / sj
        return cls(mp, sp, mj, sj, float(np.quantile(gaps, quantile)))

    def proxy_z(self, values):
        return (np.asarray(values, dtype=float) - self.proxy_mean) / self.proxy_std

    def judge_z(self, values):
        return (np.asarray(values, dtype=float) - self.judge_mean) / self.judge_std

    def gap(self, proxy, judge):
        return self.proxy_z(proxy) - self.judge_z(judge)

    def save(self, path):
        atomic_json(path, self.__dict__)

    @classmethod
    def load(cls, path):
        return cls(**read_json(path))


class GapMemory:
    def __init__(self, embeddings, gaps, group_ids, k, temperature, encoder_identity):
        self.embeddings = np.asarray(embeddings, np.float32)
        self.gaps = np.asarray(gaps, np.float32)
        self.group_ids = np.asarray(group_ids, dtype=str)
        self.k, self.temperature, self.encoder_identity = int(k), float(temperature), encoder_identity
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.gaps) or len(self.gaps) != len(self.group_ids):
            raise ValueError("Memory arrays have incompatible shapes.")
        if self.k < 1 or len(self.gaps) < self.k or self.temperature <= 0:
            raise ValueError("Memory is too small for k or invalid temperature.")
        finite(self.embeddings, "memory embeddings")
        finite(self.gaps, "memory gaps")
        if not np.allclose(np.linalg.norm(self.embeddings, axis=1), 1, atol=1e-4):
            raise ValueError("Memory embeddings must be L2-normalized.")

    def predict(self, embeddings, query_ids=None, encoder_identity=None):
        if encoder_identity is not None and encoder_identity != self.encoder_identity:
            raise ValueError("Frozen proxy encoder/reward protocol mismatch; rebuild memory.")
        q = np.asarray(embeddings, np.float32)
        finite(q, "query embeddings")
        if q.ndim != 2 or q.shape[1] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension differs from memory.")
        if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4):
            raise ValueError("Query embeddings must be L2-normalized.")
        predictions, similarities, neighbors = [], [], []
        for i, query in enumerate(q):
            valid = np.ones(len(self.embeddings), dtype=bool)
            if query_ids is not None:
                valid &= self.group_ids != str(query_ids[i])
            eligible = np.flatnonzero(valid)
            if len(eligible) < self.k:
                raise ValueError("Too few neighbors after excluding the entire query question group.")
            # Stable exact cosine top-k; no approximate-index accuracy confound.
            scores, indices = top_neighbors(query[None, :], self.embeddings[eligible], self.k)
            idx = eligible[indices[0]]
            weights = np.exp((scores[0] - scores[0].max()) / self.temperature)
            weights /= weights.sum()
            predictions.append(float(weights @ self.gaps[idx]))
            similarities.append(float(scores[0, 0]))
            neighbors.append(idx.tolist())
        return np.asarray(predictions), np.asarray(similarities), neighbors

    def extend(self, embeddings, gaps, group_ids):
        return GapMemory(np.concatenate([self.embeddings, embeddings]), np.concatenate([self.gaps, gaps]),
                         np.concatenate([self.group_ids, group_ids]), self.k, self.temperature, self.encoder_identity)

    def save(self, path):
        atomic_npz(path, embeddings=self.embeddings, gaps=self.gaps, group_ids=self.group_ids,
                   k=self.k, temperature=self.temperature, encoder_identity=self.encoder_identity)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as x:
            return cls(x["embeddings"], x["gaps"], x["group_ids"], x["k"].item(),
                       x["temperature"].item(), x["encoder_identity"].item())


def select_memory(embeddings, gaps, ids, selection_embeddings, selection_gaps, selection_ids, config, identity):
    candidates = []
    for k in config["knn"]["k_grid"]:
        for temperature in config["knn"]["temperature_grid"]:
            memory = GapMemory(embeddings, gaps, ids, k, temperature, identity)
            pred, _, _ = memory.predict(selection_embeddings, selection_ids, identity)
            candidates.append({"k": k, "temperature": temperature,
                               "selection_mse": float(np.mean((pred - selection_gaps)**2))})
    best = min(candidates, key=lambda x: (x["selection_mse"], x["k"], x["temperature"]))
    return GapMemory(embeddings, gaps, ids, best["k"], best["temperature"], identity), {"selected": best, "grid": candidates}


def corrected_reward(proxy_z, predicted_gap, mode="signed"):
    if mode not in ('signed', 'positive_only'):
        raise ValueError('Unknown correction mode.')
    predicted_gap = np.asarray(predicted_gap)
    applied = predicted_gap if mode == "signed" else np.maximum(0, predicted_gap)
    return np.asarray(proxy_z) - applied


class RidgeGap:
    """The same prediction interface as kNN; trained on actual 4B gap labels."""
    def __init__(self, coef, intercept, memory):
        self.coef, self.intercept = np.asarray(coef, float), float(intercept)
        self.gaps, self.group_ids = memory.gaps, memory.group_ids
        self.encoder_identity = memory.encoder_identity

    def predict(self, embeddings, query_ids=None, encoder_identity=None):
        if encoder_identity not in (None, self.encoder_identity):
            raise ValueError('Ridge encoder identity differs.')
        x = np.asarray(embeddings, np.float32)
        if (x.ndim != 2 or x.shape[1] != len(self.coef) or not np.isfinite(x).all() or
            not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-4)):
            raise ValueError('Invalid ridge features.')
        if query_ids is not None and set(query_ids) & set(self.group_ids):
            raise ValueError('Ridge queries overlap fitting memory.')
        return x @ self.coef+self.intercept, np.full(len(x), np.nan), [[] for _ in x]

    @classmethod
    def load(cls, folder, memory):
        folder = Path(folder)
        selected = read_json(folder/'selection.json')
        if file_sha(folder/'ridge.npz') != selected['model_sha256']:
            raise ValueError('Saved ridge coefficients changed.')
        with np.load(folder/'ridge.npz', allow_pickle=False) as data:
            if data['encoder_identity'].item() != memory.encoder_identity:
                raise ValueError('Saved ridge encoder differs.')
            return cls(data['coef'], data['intercept'].item(), memory)


def fit_ridge(memory, norm, output, config):
    """Fit/restore once during preparation, before any policy arm or final labels."""
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_limits
    from .grading import paired, unavailable
    output = Path(output)
    folder = output/'prepared/ridge'
    rows = read_jsonl(output/'prepared/selection_raw.jsonl')
    x = load_features(output/'prepared/selection_features.npz', rows, memory.encoder_identity)
    valid = paired([r['proxy_score'] for r in rows], [r['judge_score'] for r in rows])
    if not valid.any():
        unavailable(output, 'prepared/ridge', 'no valid validation grades')
        return None
    rows, x = [r for r, keep in zip(rows, valid) if keep], x[valid]
    ids = [r['id'] for r in rows]
    if set(ids) & set(memory.group_ids):
        raise ValueError('Ridge validation overlaps fitting memory.')
    y = norm.gap([r['proxy_score'] for r in rows], [r['judge_score'] for r in rows])
    dependency = digest({'memory_vectors': hashlib.sha256(memory.embeddings.tobytes()).hexdigest(),
                         'memory_gaps': hashlib.sha256(memory.gaps.tobytes()).hexdigest(),
                         'memory_ids': memory.group_ids.tolist(), 'encoder': memory.encoder_identity,
                         'normalization': norm.__dict__, 'config': config['ridge'],
                         'validation': file_sha(output/'prepared/selection_raw.jsonl'),
                         'features': file_sha(output/'prepared/selection_features.npz')})
    if (folder/'selection.json').exists():
        if read_json(folder/'selection.json')['dependency'] != dependency:
            raise ValueError('Ridge fitting inputs changed; use a new run.')
        return RidgeGap.load(folder, memory)
    counts = Counter(ids)
    weights = np.array([1/counts[q] for q in ids])
    candidates, best = [], None
    with threadpool_limits(limits=config['ridge']['cpu_threads']):
        for alpha in config['ridge']['alphas']:
            fitted = Ridge(alpha=alpha, fit_intercept=True, solver='cholesky').fit(memory.embeddings.astype(float), memory.gaps)
            mse = float(np.average((fitted.predict(x)-y)**2, weights=weights))
            candidates.append({'alpha': alpha, 'validation_question_weighted_mse': mse})
            if best is None or (mse, -alpha) < best:
                best = mse, -alpha
                model = RidgeGap(fitted.coef_, fitted.intercept_, memory)
                selected = alpha
    atomic_npz(folder/'ridge.npz', coef=model.coef, intercept=model.intercept,
               encoder_identity=memory.encoder_identity)
    atomic_json(folder/'selection.json', {'dependency': dependency, 'alpha': selected, 'candidates': candidates,
        'model_sha256': file_sha(folder/'ridge.npz'), 'memory_answers': len(memory.gaps),
        'memory_questions': len(set(memory.group_ids)), 'validation_answers': len(rows),
        'validation_excluded': int((~valid).sum()), 'test_used': False, 'refit_on_validation': False,
        'target': 'actual normalized proxy minus 4B judge gap', 'new_judge_calls': 0})
    return model
