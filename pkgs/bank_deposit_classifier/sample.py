from collections import Counter
import math

import pandas as pd


def upsample_minority_class(data, outcome, p_minority):
    def check_p_minority_bounds(p_minority):
        if (p_minority > 1) or (p_minority < 0):
            msg = 'Proportion out of bounds! p_minority must be between ' \
                    f'0 and 1, but value passed was {p_minority}.'
            raise ValueError(msg)

    def check_outcome_binary(data, outcome):
        outcome_counts = Counter(data[outcome])
        n_outcomes = len(outcome_counts.keys())
        if n_outcomes != 2:
            msg = 'Binary outcome expected but specified outcome ' \
                    f'has {n_outcomes} classes'
            raise ValueError(msg)

    check_p_minority_bounds(p_minority)
    check_outcome_binary(data, outcome)

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
