\# K-Detect — SciHRA Experiment Log



\## 1. Dataset



\- Dataset: SciHRA-Detect

\- Total articles: 3,000

\- Classes: Human-generated (HGT), AI-revised (ART), AI-generated (AGT)

\- Samples: 9,000 abstracts

\- Train: 2,400 articles / 7,200 abstracts

\- Validation: 300 articles / 900 abstracts

\- Test: 300 articles / 900 abstracts

\- Split strategy: Article-level grouped splitting

\- Group overlap: Zero



\## 2. Models evaluated



1\. Word TF-IDF + Logistic Regression

2\. Character TF-IDF + Linear SVM

3\. Equal-weight ensemble of the word and calibrated character models



\## 3. Final test results



| Model | Accuracy | Macro F1 |

|---|---:|---:|

| Word TF-IDF + Logistic Regression | 93.67% | 93.63% |

| Calibrated Character SVM | 95.78% | 95.78% |

| Equal-weight ensemble | 96.00% | 96.00% |



The ensemble achieved the highest observed test performance, but its improvement over Character SVM was small.



\## 4. Paired article-level bootstrap



\- Bootstrap iterations: 5,000

\- Resampling unit: Article

\- Resampling strategy: Sampling articles with replacement, retaining all three abstracts per sampled article

\- Random seed: 42

\- Comparison: Equal-weight ensemble minus calibrated Character SVM



\### Results



| Metric | Observed difference | 95% confidence interval |

|---|---:|---:|

| Accuracy | +0.2222 percentage points | \[-0.5556, +1.0000] |

| Macro F1 | +0.2208 percentage points | \[-0.5570, +0.9991] |



Bootstrap proportion with positive difference:



\- Macro F1: 70.44%

\- Accuracy: 66.46%



\### Interpretation



The observed ensemble advantage is small. Both confidence intervals include zero, so the paired bootstrap does not establish a reliable performance improvement over the calibrated Character SVM.



The proportions of positive bootstrap differences are descriptive statistics and should not be interpreted as formal p-values.



\## 5. Current decision



\- Retain Character SVM as the simpler strong baseline.

\- Retain the ensemble as an experimental candidate.

\- Do not claim that the ensemble is statistically superior.

\- Do not tune model weights using the final test set.

\- Seek independent external evaluation before selecting a production model.



\## 6. Limitations



\- Evaluation uses one academic abstract benchmark.

\- The dataset uses GPT-4o for AI-generated variants.

\- Source and discipline distributions are confounded.

\- The random article-level split does not establish generalization to unseen sources or disciplines.

\- Abstracts are not equivalent to complete academic papers.

\- Performance does not establish reliability for real-world authorship attribution.



\## 7. Next research priorities



1\. External academic datasets.

2\. Source-held-out and domain-held-out evaluation.

3\. Robustness against paraphrasing and human editing.

4\. Evaluation across different AI generation models.

5\. Fairness analysis across disciplines and writing proficiency groups.

6\. Calibration and uncertainty analysis.

7\. Independent validation before deployment.
