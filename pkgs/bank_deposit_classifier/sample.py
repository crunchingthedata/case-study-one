from collections import Counter
import math

import pandas as pd


def upsample_minority_class(data, outcome, p_minority):
    outcome_counts = Counter(data[outcome])
    majority_class, majority_count = outcome_counts.most_common()[0]
    minority_class, minority_count = outcome_counts.most_common()[-1]
    desired_total_count = math.ceil(majority_count/(1-p_minority))
    n_samples = desired_total_count - majority_count - minority_count
    samples = data \
        .loc[data[outcome] == minority_class] \
        .sample(n_samples, replace=True)
    upsampled_data = pd.concat([data, samples])
    return upsampled_data
