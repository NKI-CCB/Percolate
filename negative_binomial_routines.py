import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib.embedded import RRuntimeError
import pandas as pd
from copy import deepcopy
import numpy as np
rpy2.robjects.numpy2ri.activate()

def compute_dispersion(X):
    n_samples, n_genes = X.shape
    X_norm = deepcopy(X).values
    
    try:
        deseq = importr('DESeq2')
    except RRuntimeError:
        print('WARNING: DESeq is not installed', flush=True)
        return False

    # Add 1 to the highest expressed genes ; hack to make DESeq not fail.
    X_norm[:,np.argsort(np.sum(X_norm > 0, axis=0))[-1]] = X_norm[:,np.argsort(np.sum(X_norm > 0, axis=0))[-1]].clip(1)
    
    robjects.r.assign("count_data", X_norm.astype(int).transpose())
    robjects.r.assign("name_samples", np.array(X.index.get_level_values(0)).astype(str))
    
    robjects.r('''
        dds <- DESeqDataSetFromMatrix(countData = count_data, colData = name_samples, design = ~1);
        dds <- estimateSizeFactors(dds);
        dds <- estimateDispersions(dds);
        a <- dispersions(dds)
    ''')
    
    return pd.DataFrame(
        np.array(robjects.r['a']),
        index=X.columns,
        columns=['dispersion']
    )