# Evaluation summary

Golden-set size: **200**

| system | intent_accuracy | intent_macro_f1 | escalation_f1 | false_auto_handle_rate | recall@5 | mrr | judge_overall_mean | hallucination_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0.110 | 0.022 | 0.718 | 0.000 | 0.000 | 0.000 | 2.000 | 0.000 |
| tfidf_baseline | 0.645 | 0.656 | 0.715 | 0.339 | 0.640 | 0.640 | 2.805 | 0.000 |
| proposed | 0.645 | 0.660 | 0.752 | 0.241 | 0.645 | 0.642 | 2.845 | 0.000 |

Full JSON: `reports/results.json`. Confusion matrix: `reports/confusion_matrix.png`.
Failure analysis: `reports/failure_analysis.md`.
Headline caveats: `reports/headline_number.md`.
