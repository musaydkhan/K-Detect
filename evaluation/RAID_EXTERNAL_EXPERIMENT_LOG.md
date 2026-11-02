\# K-Detect: RAID External Evaluation Experiment Log



\## 1. Experiment Objective



This experiment evaluates how well classifiers trained on the SciHRA-Detect dataset generalize to AI-generated scientific abstracts from an external benchmark.



The primary objective is to investigate generator-specific performance differences and identify potential failures when models encounter AI-generated text from generators not represented in their training data.



This is an external evaluation using a subset of the RAID dataset, not an evaluation on the official hidden RAID test set.



\## 2. Dataset



\*\*Dataset:\*\* RAID (A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors)



\*\*Paper:\*\* Dugan et al. (2024), \*RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors\*, ACL 2024.



\*\*Dataset source:\*\* https://huggingface.co/datasets/liamdugan/raid



\*\*Subset used:\*\* `train\_none.csv`



The downloaded subset contained 467,985 rows. After extracting scientific abstracts, 61,810 records were retained.



The extracted records contained:



\- 1,766 unique human source documents.

\- 11 AI generator variants.

\- Two decoding conditions: greedy and sampling.

\- 35 records per human source across the complete extracted dataset.



For the primary external evaluation, the analysis used the 22 standard generator/decoding conditions, excluding additional repetition-penalty records.



Each condition contained 1,766 human references and 1,766 corresponding AI-generated abstracts.



\### 2.1 Generators



The 11 generator variants were:



1\. GPT-2

2\. GPT-3

3\. GPT-4

4\. ChatGPT

5\. MPT

6\. MPT-chat

7\. Mistral

8\. Mistral-chat

9\. Llama-chat

10\. Cohere

11\. Cohere-chat



\### 2.2 Prompt characteristics



RAID's abstract-generation prompts used the paper title as the principal input.



The chat prompt template was:



`Write the abstract for the academic paper titled "{title}".`



The non-chat template used a continuation-style instruction referring to the full text of an abstract for a paper with the specified title.



The original human abstract was not supplied as the generation input.



Therefore, this experiment evaluates title-conditioned generation rather than direct rewriting of the original human abstract.



\### 2.3 Dataset independence check



An exact normalized-text comparison between the RAID human references and SciHRA abstracts found no exact matches.



A character n-gram similarity analysis found:



\- Maximum similarity: 0.4085

\- Similarity >= 0.70: 0 pairs

\- Similarity >= 0.80: 0 pairs

\- Similarity >= 0.90: 0 pairs

\- Similarity >= 0.95: 0 pairs



These checks reduce concern about direct textual overlap but do not prove complete source independence. Character n-gram similarity is not a comprehensive source-independence test. The analysis does not rule out paraphrases, shared source publications, or overlap in paper titles.



\## 3. Classifiers



Three distinct classification approaches were evaluated.



\### 3.1 TF-IDF Logistic Regression



\- Word-level TF-IDF features.

\- Word unigrams and bigrams.

\- Vocabulary size: 50,000.

\- Logistic Regression classifier.

\- Trained on SciHRA's three classes.



\### 3.2 Character-level SVM



\- Character-level TF-IDF features.

\- Character n-grams: 3–5.

\- Linear SVM with probability calibration.

\- Trained on SciHRA's three classes.



\### 3.3 Equal-weight Ensemble



The ensemble averaged the AI probabilities from the word-based Logistic Regression and character-level SVM.



The ensemble was an experimental combination of the two classifiers.



The separate `Ensemble Word Model` result in the prediction analysis is identical to TF-IDF Logistic Regression and is not an independent classifier. It is excluded from comparisons involving distinct classifiers.



\### 3.4 Probability interpretation



The SciHRA models use three classes:



\- Class 0: Human-generated

\- Class 1: AI-revised

\- Class 2: AI-generated



For external binary evaluation:



`AI score = P(class 1) + P(class 2)`



`Human score = P(class 0)`



An abstract was classified as AI-generated when its AI score was at least 0.50.



The SciHRA-derived probabilities were applied to RAID without external recalibration. Consequently, the scores should not be interpreted as calibrated probabilities of AI authorship on RAID.



\## 4. Evaluation Procedure



The evaluation paired each human reference with one AI-generated abstract from the corresponding source and generation condition.



For each generator and decoding condition:



\- 1,766 human references were evaluated.

\- 1,766 AI-generated abstracts were evaluated.

\- Total: 3,532 observations per condition.



The human references were reused across generator and decoding conditions.



The primary metrics were:



\- Accuracy

\- Balanced accuracy

\- Precision

\- Recall

\- F1-score

\- ROC-AUC

\- False-positive rate (FPR)

\- False-negative rate (FNR)



A classification threshold of 0.50 was used.



\### 4.1 Aggregation



Two types of aggregation were used:



1\. \*\*Mean-of-condition metrics:\*\* Metrics calculated separately for each generator/decoding condition and then averaged.

2\. \*\*Pooled metrics:\*\* Metrics calculated over the complete prediction set for each classifier.



These values are not interchangeable. In particular, F1-score is nonlinear, so the mean of condition-level F1 values can differ from the pooled F1-score.



The pooled evaluation contains balanced human and AI-generated observations because each source-condition contributes one human reference and one AI-generated abstract.



\### 4.2 Paired source-level bootstrap comparison



A paired source-level bootstrap with 5,000 iterations was conducted.



The 1,766 unique `source\_id` values were used as the resampling units. Each bootstrap sample resampled source IDs and retained the associated paired human and AI-generated observations across the evaluation conditions.



This approach accounts for the reuse of human references across generator and decoding conditions by keeping observations from the same source together during resampling.



The resulting confidence intervals describe uncertainty conditional on this RAID subset and evaluation pipeline. They do not establish generalization to every possible generator or scientific-writing population.



The reported `bootstrap\_positive\_percentage` indicates the percentage of bootstrap iterations in which the observed comparison direction was positive. It is a descriptive bootstrap statistic, not a p-value.



\## 5. Overall External Evaluation



\### 5.1 Mean-of-condition results



| Classifier | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |

|---|---:|---:|---:|---:|---:|---:|---:|

| TF-IDF Logistic Regression | 0.6463 | 0.6621 | 0.3435 | 0.4071 | 0.6378 | 0.0510 | 0.6565 |

| Character SVM | 0.5539 | 0.5113 | 0.2578 | 0.3265 | 0.5664 | 0.1501 | 0.7422 |

| Equal-weight Ensemble | 0.5826 | 0.5495 | 0.2706 | 0.3390 | 0.6106 | 0.1053 | 0.7294 |



The word-based Logistic Regression performed best overall.



The equal-weight ensemble did not improve upon the word-based classifier. The character classifier also showed weaker external performance.



\### 5.2 Pooled results



| Classifier | Accuracy | Balanced Accuracy | Pooled F1 |

|---|---:|---:|---:|

| TF-IDF Logistic Regression | 0.6463 | 0.6463 | 0.4927 |

| Character SVM | 0.5539 | 0.5539 | 0.3662 |

| Equal-weight Ensemble | 0.5826 | 0.5826 | 0.3934 |



The pooled evaluation was balanced, with one human and one AI-generated abstract per source-condition. Therefore, accuracy and balanced accuracy were equal.



\### 5.3 Paired bootstrap comparisons



| Comparison | Accuracy Difference | 95% CI | F1 Difference | 95% CI |

|---|---:|---|---:|---|

| TF-IDF minus Character SVM | +0.0924 | \[0.0842, 0.1005] | +0.1265 | \[0.1196, 0.1332] |

| TF-IDF minus Ensemble | +0.0636 | \[0.0572, 0.0704] | +0.0993 | \[0.0937, 0.1049] |

| Character SVM minus Ensemble | -0.0288 | \[-0.0340, -0.0236] | -0.0271 | \[-0.0308, -0.0233] |



All reported intervals exclude zero.



These comparisons favor TF-IDF Logistic Regression over the other two approaches in this particular external evaluation. No multiple-comparison correction was applied.



The confidence intervals should be interpreted in the context of the selected RAID subset and the specified resampling procedure, rather than as evidence of universal model superiority.



\## 6. Generator-Specific Analysis



The following results average the greedy and sampling conditions for each generator.



\### 6.1 TF-IDF Logistic Regression



| Generator | Mean F1 | Mean ROC-AUC | Mean FPR | Mean FNR | Mean Score Difference |

|---|---:|---:|---:|---:|---:|

| GPT-4 | 0.9605 | 0.9935 | 0.0510 | 0.0289 | +0.5689 |

| MPT-chat | 0.8932 | 0.9617 | 0.0510 | 0.1518 | +0.4635 |

| ChatGPT | 0.8419 | 0.9451 | 0.0510 | 0.2358 | +0.4073 |

| Llama-chat | 0.5630 | 0.8059 | 0.0510 | 0.5875 | +0.2262 |

| Mistral-chat | 0.4920 | 0.7635 | 0.0510 | 0.6537 | +0.1869 |

| Cohere-chat | 0.3528 | 0.6559 | 0.0510 | 0.7749 | +0.1103 |

| Cohere | 0.1954 | 0.5677 | 0.0510 | 0.8862 | +0.0420 |

| GPT-2 | 0.0752 | 0.3461 | 0.0510 | 0.9584 | -0.0617 |

| Mistral | 0.0380 | 0.3441 | 0.0510 | 0.9796 | -0.0686 |

| GPT-3 | 0.0323 | 0.3209 | 0.0510 | 0.9827 | -0.0788 |

| MPT | 0.0343 | 0.3118 | 0.0510 | 0.9816 | -0.0815 |



\### 6.2 Character SVM



| Generator | Mean F1 | Mean ROC-AUC | Mean FPR | Mean FNR | Mean Score Difference |

|---|---:|---:|---:|---:|---:|

| GPT-4 | 0.8058 | 0.8910 | 0.1501 | 0.2237 | +0.5539 |

| MPT-chat | 0.6371 | 0.7568 | 0.1501 | 0.4623 | +0.3308 |

| ChatGPT | 0.5507 | 0.7025 | 0.1501 | 0.5609 | +0.2440 |

| Llama-chat | 0.4358 | 0.6415 | 0.1501 | 0.6787 | +0.1585 |

| Cohere | 0.2909 | 0.5743 | 0.1501 | 0.8041 | +0.0597 |

| Cohere-chat | 0.3025 | 0.5313 | 0.1501 | 0.7950 | +0.0455 |

| Mistral-chat | 0.2594 | 0.5018 | 0.1501 | 0.8259 | +0.0167 |

| GPT-3 | 0.0753 | 0.4543 | 0.1501 | 0.9550 | -0.0572 |

| GPT-2 | 0.0918 | 0.4314 | 0.1501 | 0.9437 | -0.0423 |

| Mistral | 0.0749 | 0.3920 | 0.1501 | 0.9553 | -0.0821 |

| MPT | 0.0673 | 0.3533 | 0.1501 | 0.9598 | -0.1000 |



\### 6.3 Equal-weight Ensemble



| Generator | Mean F1 | Mean ROC-AUC | Mean FPR | Mean FNR | Mean Score Difference |

|---|---:|---:|---:|---:|---:|

| GPT-4 | 0.8662 | 0.9608 | 0.1053 | 0.1554 | +0.5614 |

| MPT-chat | 0.7219 | 0.8903 | 0.1053 | 0.3757 | +0.3972 |

| ChatGPT | 0.6320 | 0.8551 | 0.1053 | 0.4878 | +0.3257 |

| Llama-chat | 0.4634 | 0.7312 | 0.1053 | 0.6653 | +0.1924 |

| Mistral-chat | 0.3000 | 0.6538 | 0.1053 | 0.8018 | +0.1018 |

| Cohere-chat | 0.2944 | 0.6030 | 0.1053 | 0.8092 | +0.0779 |

| Cohere | 0.2484 | 0.5771 | 0.1053 | 0.8431 | +0.0509 |

| GPT-2 | 0.0649 | 0.3848 | 0.1053 | 0.9626 | -0.0520 |

| GPT-3 | 0.0406 | 0.3831 | 0.1053 | 0.9771 | -0.0680 |

| Mistral | 0.0485 | 0.3591 | 0.1053 | 0.9725 | -0.0754 |

| MPT | 0.0484 | 0.3183 | 0.1053 | 0.9725 | -0.0907 |



\### 6.4 Interpretation



The external results show substantial generator-specific heterogeneity.



GPT-4, MPT-chat, and ChatGPT were comparatively distinguishable, particularly with the word-based classifier.



Several non-chat generator variants, including GPT-2, GPT-3, Mistral, and MPT, showed very poor performance. Their mean ROC-AUC values were below 0.5 across all three classifiers.



A ROC-AUC below 0.5 indicates that the model tended to rank generated samples below human references in these conditions. It does not establish that reversing the classifier's predictions would create a reliable detector.



The mean score difference is defined as:



`Mean generated AI score - Mean human AI score`



Negative values indicate that generated abstracts received lower mean AI scores than human references.



The word-based classifier maintained a relatively low mean false-positive rate but showed very high false-negative rates for several generators. This demonstrates that a low false-positive rate does not guarantee reliable detection across different generators.



These observations are descriptive. They do not establish that chat fine-tuning or decoding strategy alone caused the differences, because the generator variants are distinct systems.



\## 7. GPT-2 Sampling Score Inspection



A targeted inspection was conducted for TF-IDF Logistic Regression on GPT-2 sampling.



| Statistic | Human references | GPT-2 sampling |

|---|---:|---:|

| Count | 1,766 | 1,766 |

| Mean AI score | 0.2265 | 0.1065 |

| Standard deviation | 0.1437 | 0.1111 |

| Median | 0.1962 | 0.0679 |

| 25th percentile | 0.1162 | 0.0312 |

| 75th percentile | 0.3094 | 0.1422 |

| Maximum | 0.8462 | 0.6994 |



At the 0.50 classification threshold:



\- Human references classified as AI: 90

\- GPT-2 generations classified as AI: 29



The generated samples received a lower mean AI score than the human references.



This provides a concrete example of score inversion and helps explain the low ROC-AUC for GPT-2 sampling.



\## 8. Interpretation and Main Findings



The external evaluation supports the following findings:



1\. Internal SciHRA performance does not transfer uniformly to RAID.

2\. Generator identity is associated with substantial differences in detection performance.

3\. Some generator conditions produce reversed score rankings.

4\. TF-IDF Logistic Regression outperformed the character SVM and equal-weight ensemble in this external evaluation.

5\. Equal-weight averaging did not improve overall external performance.

6\. A relatively low false-positive rate can coexist with very high false-negative rates.

7\. High internal test performance should not be interpreted as evidence of universal AI-text detection capability.



The results support treating AI-text detection as a robustness and generalization problem rather than relying on a single accuracy score.



\## 9. Limitations



\### Dataset limitations



\- The experiment used a RAID training subset, not the official hidden RAID test set.

\- The extracted subset contains scientific abstracts, but the original RAID benchmark covers multiple text domains.

\- Human references were reused across generator conditions.

\- The benchmark's generator and prompting conditions may not represent current or future generation systems.

\- Exact and near-duplicate checks cannot guarantee complete independence between datasets.



\### Model limitations



\- Models were trained on SciHRA and evaluated on RAID without external retraining.

\- SciHRA class probabilities were not recalibrated for RAID.

\- The 0.50 threshold was retained from the original binary scoring procedure.

\- Equal-weight averaging assumes that model scores are sufficiently comparable, which has not been established through external calibration.

\- The word-based and character-based models may learn dataset-specific stylistic and lexical associations rather than general AI-generation characteristics.



\### Statistical limitations



\- Bootstrap intervals are conditional on the selected RAID subset and evaluation design.

\- No multiple-comparison correction was applied to the paired classifier comparisons.

\- Generator-level summary values average greedy and sampling conditions; they are not independent experiments.

\- No generator-specific bootstrap confidence intervals were calculated.

\- Differences between generator variants should not be interpreted as causal effects of chat tuning or decoding strategy.



\### Practical limitations



The results do not establish that K-Detect can reliably identify AI authorship in arbitrary academic documents.



No automated detection score should be used as proof of misconduct or authorship.



\## 10. Reproducibility and Output Files



The main scripts and generated files are:



| File | Purpose |

|---|---|

| `evaluate\_raid\_external.py` | External RAID evaluation |

| `analyze\_raid\_generators.py` | Generator-specific metric analysis |

| `bootstrap\_raid\_comparison.py` | Paired source-level bootstrap comparison |

| `test\_raid\_inference.py` | Model inference smoke test |

| `raid\_external\_predictions.csv` | Per-source prediction scores |

| `raid\_external\_evaluation\_summary.csv` | Generator/decoding condition-level metrics |

| `raid\_generator\_analysis.csv` | Generator-specific metrics |

| `raid\_paired\_bootstrap\_results.csv` | Bootstrap comparison results |

| `raid\_scihra\_near\_duplicates.csv` | Near-duplicate comparison results |



The downloaded RAID data and derived prediction files should not be redistributed until dataset and source-text licensing conditions have been reviewed.



For reproducibility, the repository should also document the Python version, dependency versions, model artifact locations, and commands required to reproduce the analyses.



\## 11. Overall Conclusion



The RAID external evaluation reveals a substantial generalization gap between internal SciHRA performance and performance on an external benchmark subset.



Although the TF-IDF Logistic Regression classifier performed best among the evaluated approaches, its performance varied sharply across generator conditions. Several generators produced score distributions that were poorly separated from, or reversed relative to, the human reference distribution.



The equal-weight ensemble did not provide a consistent improvement.



These findings suggest that detector performance depends strongly on the generation system and evaluation domain. They reinforce the need for external validation, generator-diverse benchmarks, careful threshold selection, and transparent reporting of false positives and false negatives.



\*\*Research framing:\*\* K-Detect is a research-oriented framework for evaluating the robustness and generalization of AI-generated scientific-text detection approaches. The current results are exploratory benchmark findings, not evidence of a universally reliable authorship detector.



\---



\*\*Status:\*\* RAID external evaluation documented. Further experiments should be undertaken only if a specific unresolved methodological issue is identified.
