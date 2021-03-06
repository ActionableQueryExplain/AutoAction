# AutoAction

Automated Generation of Actionable Query Explanations

Full technicial report is available through [`autoaction_tech_report.pdf`][1]


# Setup
`pip install -r requirements.txt`


# Datasets

Datasets used in this experiments are available in [`./datasets`][2]

For synthetic credit1M dataset, refer to https://tinyurl.com/2sm97rwy

Synthetic datasets can be generated by executing `./scripts/_dataset_generator.py n_rows n_groups_1 n_groups_2 test.csv`, where `n_groups_1` and `n_groups_2` are number of groups for two group attributes, respectively.


# Demo
The demo for Autoaction is available on [`demo.ipynb`][3]

Default dataset configuration is from `{adult, student, credit100k, credit1m}`

To add own customized configuration for other datasets, execute

```
from dataConfig import dataLoader
dl = dataLoader()
dataset = dl.customized_loader(
        group_attrs,
        other_attrs,
        actionable_attrs,
        agg_attr,
        fit_attrs,
        path,
        action_constraints,
        instance_constraints,
        appro_instance_constraints,
        conditional_constraints,
        appro_conditional_constraints,
        name)
```

# Experiments

The experiments executed for the paper is available on [`experiments.ipynb`][4]

`trials` can be adjusted to specify the number of trials. Set the time budget(s) in `time`.

# Results

Original experimential results for the paper is in [`./results`][5]

Charts in the paper can be reproduced via [`results.ipynb`][6]

[1]: https://github.com/ActionableQueryExplain/AutoAction/blob/main/autoaction_tech_report.pdf
[2]: https://github.com/ActionableQueryExplain/AutoAction/tree/main/datasets
[3]: https://github.com/ActionableQueryExplain/AutoAction/blob/main/demo.ipynb
[4]: https://github.com/ActionableQueryExplain/AutoAction/blob/main/experiments.ipynb
[5]: https://github.com/ActionableQueryExplain/AutoAction/tree/main/results
[6]: https://github.com/ActionableQueryExplain/AutoAction/blob/main/results.ipynb

