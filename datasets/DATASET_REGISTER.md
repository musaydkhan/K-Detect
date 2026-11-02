

\# K-Detect Dataset Register



This document tracks candidate datasets, their intended use, and licensing status.



\## Candidate datasets



| Dataset | Intended use | Status |

|---|---|---|

| PMC Open Access Subset | Primary source of human-written academic text | Candidate - filter by article license |

| RAID | AI text training and external robustness evaluation | Candidate - investigate dataset terms and size |

| HC3 | Supplementary human vs ChatGPT experiments | Candidate - filter source-level permissions |

| TuringBench | Historical generator diversity benchmark | Candidate - verify underlying data rights |



\## Planned original dataset



K-Detect will develop an academic text corpus containing:



\- Human-written academic passages

\- AI-generated academic passages

\- Human-edited AI-generated passages



Each passage must retain its source document, discipline, generation model (if applicable), editing history, and applicable license information.



\## Data collection rules



1\. Never assume free access means permission to reuse.

2\. Verify licenses at the source-document level where necessary.

3\. Record the original dataset and source for every passage.

4\. Keep human-edited AI text as a separate category.

5\. Prevent source-document overlap between training, validation, and test sets.

6\. Do not publish copyrighted source text without permission.

7\. Do not use test data for training or model selection.



\## Current decision



No dataset has been downloaded or approved for training yet.



All candidate datasets require licensing and suitability review before use.
