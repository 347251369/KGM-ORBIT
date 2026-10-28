import csv
import numpy as np
import pandas as pd

def read_table(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    if len(header) < 2 or len(set(header)) != len(header) or any(not h.strip() for h in header[1:]):
        raise ValueError('CSV requires a sample ID column and unique data columns')
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if frame.iloc[:, 0].str.strip().eq('').any():
        raise ValueError('Sample IDs cannot be empty')
    frame = frame.set_index(frame.columns[0])
    return frame.replace(['', 'NA', 'NaN', 'nan', 'N/A', 'null'], np.nan)


def read_expression(path):
    frame = read_table(path)
    if frame.empty or frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError('Expression requires unique sample IDs and gene columns')
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame.astype(float)

def map_labels(labels, mapping):
    result = pd.Series(labels).map(mapping)
    if result.isna().any() or not result.isin([0, 1, 2]).all():
        raise ValueError('Every clinical label needs an explicit 0/1/2 mapping')
    return result.astype(int)

def map_probes(expression, mapping, *, aggregation):
    if aggregation not in {'mean', 'median'}:
        raise ValueError('Choose mean or median probe aggregation')
    genes = [mapping.get(str(p)) for p in expression.columns]
    keep = [i for (i, gene) in enumerate(genes) if gene]
    x = expression.iloc[:, keep].copy()
    x.columns = [genes[i] for i in keep]
    return x.T.groupby(level=0, sort=True).agg(aggregation).T

class GeneStandardizer:

    def fit(self, expression):
        x = expression.astype(float)
        if x.empty or x.columns.has_duplicates or np.isinf(x.to_numpy()).any():
            raise ValueError('Invalid training expression')
        median = x.median()
        filled = x.fillna(median)
        scale = filled.std(ddof=0)
        keep = median.notna() & scale.notna() & (scale > 0)
        if not keep.any():
            raise ValueError('No usable training genes')
        self.genes = list(x.columns[keep])
        self.median = median[keep].to_numpy()
        self.center = filled[self.genes].mean().to_numpy()
        self.scale = scale[keep].to_numpy()
        return self

    def transform(self, expression):
        x = expression.reindex(columns=self.genes).to_numpy(float)
        if np.isinf(x).any():
            raise ValueError('Infinite expression')
        observed = np.isfinite(x)
        filled = np.where(observed, x, self.median)
        return ((filled - self.center) / self.scale, observed)

    def to_dict(self):
        return dict(genes=self.genes,
            median=self.median.tolist(),
            center=self.center.tolist(),
            scale=self.scale.tolist())

    @classmethod
    def from_dict(cls, value):
        obj = cls()
        obj.genes = value['genes']
        for key in ['median', 'center', 'scale']:
            setattr(obj, key, np.asarray(value[key], float))
        if (not obj.genes or len(set(obj.genes)) != len(obj.genes)
                or any(not isinstance(g, str) or not g for g in obj.genes)
                or any(getattr(obj, k).shape != (len(obj.genes),)
                       or not np.isfinite(getattr(obj, k)).all() for k in ['median', 'center', 'scale'])
                or (obj.scale <= 0).any()):
            raise ValueError('Invalid gene preprocessing statistics')
        return obj
