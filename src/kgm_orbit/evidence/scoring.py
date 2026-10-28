import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from .core import ContrastiveHead

class PairScorer:

    def __init__(self, encoder, head):
        self.encoder = encoder
        self.head = head

    def features(self, sources, claims):
        if len(sources) != len(claims):
            raise ValueError('Pair count mismatch')
        a = np.asarray(self.encoder.encode(sources, normalize_embeddings=True))
        b = np.asarray(self.encoder.encode(claims, normalize_embeddings=True))
        return np.concatenate([a, b, np.abs(a - b), a * b], axis=1)

    def __call__(self, source, claim):
        return float(expit(self.head.score(self.features([source], [claim]))[0]))

    def score_pairs(self, sources, claims):
        return expit(self.head.score(self.features(sources, claims)))

    def margin(self, source, positive, negative):
        if len(positive) != 2 or len(negative) != 2:
            raise ValueError('Two positive/negative paraphrase pairs required')
        scores = self.score_pairs([source] * 4, list(positive) + list(negative))
        return float(np.mean(scores[:2] - scores[2:]))


class IterativeScorer:
    def __init__(self, encoder, validation_claims, *, l2=0.1):
        if not np.isfinite(l2) or l2 < 0:
            raise ValueError('Nonnegative finite L2 required')
        self.encoder = encoder
        self.validation = list(validation_claims)
        if not self.validation or {r['label'] for r in self.validation} != {0, 1}:
            raise ValueError('Verified positive and negative validation claims required')
        self.l2 = l2
        self.groups = {}
        self.scorer = None
        self.history = []

    def update(self, records):
        from .workflow import _article_key
        held_out = {_article_key(r['source_id']) for r in self.validation}
        added = 0
        for record in records:
            if _article_key(record['source_id']) in held_out:
                continue
            for positive in record['positive']:
                key = (record['source_id'], record['source'], positive, tuple(record['negative']))
                if key not in self.groups:
                    self.groups[key] = dict(source_id=record['source_id'], source=record['source'],
                        positive=positive, negatives=record['negative'])
                    added += 1
        if added:
            self.scorer, self.tau = fit_scorer(self.encoder, self.groups.values(), self.validation, l2=self.l2)
            self.history.append(dict(training_groups=len(self.groups), tau=self.tau))
        if self.scorer is None:
            raise ValueError('No eligible ESA training claims outside validation sources')
        return self.tau

    def margin(self, source, positive, negative):
        if self.scorer is None:
            raise ValueError('Train ESA on extracted claims before scoring')
        return self.scorer.margin(source, positive, negative)


def fit_scorer(encoder, training_groups, validation_claims, *, l2=0.1):
    training_groups, validation_claims = list(training_groups), list(validation_claims)
    if not training_groups or not validation_claims:
        raise ValueError('Training groups and validation claims required')
    train_sources = {r['source_id'] for r in training_groups}
    validation_sources = {r['source_id'] for r in validation_claims}
    if train_sources & validation_sources:
        raise ValueError('Training and validation sources overlap')
    sources, texts, groups = [], [], []
    for record in training_groups:
        examples = [record['positive'], *record['negatives']]
        if len(examples) < 2 or not all(isinstance(t, str) and t.strip() for t in examples):
            raise ValueError('A positive claim and nonempty negative claims required')
        start = len(texts)
        groups.append(list(range(start, start + len(examples))))
        texts.extend(examples)
        sources.extend([record['source']] * len(examples))
    scorer = PairScorer(encoder, ContrastiveHead(l2=l2))
    scorer.head.fit(scorer.features(sources, texts), groups)
    margins = [scorer.margin(r['source'], r['positive'], r['negative'])
               for r in validation_claims]
    labels = [r['label'] for r in validation_claims]
    tau = choose_threshold(labels, margins, nonnegative=True)
    return scorer, tau

def choose_threshold(labels, scores, *, nonnegative):
    y = np.asarray(labels)
    s = np.asarray(scores, float)
    if y.shape != s.shape or set(y.tolist()) != {0, 1} or (not np.isfinite(s).all()):
        raise ValueError('Aligned binary validation labels and finite scores required')
    unique = np.unique(s)
    values = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2, unique[-1]]
    if nonnegative:
        values = np.unique(np.r_[0, values[values >= 0]])
    return float(max(values, key=lambda t: (f1_score(y, s > t, zero_division=0), t)))
