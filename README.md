# K-Detect

**A research framework for evaluating AI-generated scientific writing detection.**

K-Detect is an independent research project investigating how reliably machine-learning methods distinguish human-written, AI-revised, and AI-generated scientific text.

The project combines linguistic analysis, traditional machine-learning classifiers, and systematic evaluation to investigate detection performance, cross-dataset generalization, and robustness across different text-generation systems.

K-Detect is an experimental research platform, not a definitive authorship verification system.

## Research Objectives

The project focuses on:

- Evaluating machine-learning approaches to AI-generated scientific text detection.
- Comparing word-level and character-level text representations.
- Distinguishing human-written, AI-revised, and AI-generated abstracts.
- Investigating cross-dataset performance and generator-specific variation.
- Measuring false positives, false negatives, and classification uncertainty.
- Examining the limitations of automated AI-text detection.

## Research Components

### 1. Scientific Writing Classification

Three-class classification experiments using the SciHRA-Detect dataset:

- **Human-generated (HGT):** Human-written scientific abstracts.
- **AI-revised (ART):** Human-written abstracts revised using AI.
- **AI-generated (AGT):** AI-generated scientific abstracts.

### 2. Machine Learning Models

The experimental pipeline includes:

- TF-IDF with Logistic Regression.
- Character-level TF-IDF with Linear Support Vector Classification.
- Probability calibration for character-level classification.
- Experimental probability-based ensemble evaluation.

### 3. Evaluation and Robustness Analysis

Experiments include:

- Article-level grouped train, validation, and test splitting.
- Accuracy, precision, recall, macro F1, and confusion-matrix analysis.
- Bootstrap confidence intervals.
- Paired model comparisons.
- Feature ablation and feature-frequency analysis.
- Scientific-domain performance analysis.
- External evaluation using a subset of the RAID benchmark.
- Generator-specific and decoding-condition comparisons.

## Selected Experimental Findings

### SciHRA-Detect

On the held-out SciHRA test split:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF Logistic Regression | 93.67% | 93.63% |
| Character-level SVM | 95.78% | 95.78% |
| Equal-weight ensemble | 96.00% | 96.00% |

The ensemble's improvement over the character-level SVM was small. Paired article-level bootstrap confidence intervals included zero, so the experiments did not establish a statistically conclusive improvement.

### RAID External Evaluation

The trained SciHRA classifiers were evaluated on a non-adversarial subset of the RAID benchmark. This was an external evaluation subset, not the official hidden RAID test set.

The results revealed substantial differences in detection performance across generators. Models that performed well on some generator families performed poorly on others.

The equal-weight ensemble did not consistently improve external performance over the word-level Logistic Regression baseline.

These findings highlight the difficulty of transferring AI-text detection performance across datasets and generation systems.

**Important:** These results describe the specific datasets, experimental splits, and evaluation conditions used in this project. They should not be interpreted as estimates of universal real-world detection accuracy.

## Application Features

The Streamlit application provides an interface for academic writing analysis, including:

- TXT and DOCX document input.
- Text processing and linguistic analysis.
- Descriptive writing statistics.
- Downloadable analysis reports.

The research classifiers and evaluation pipelines are maintained as separate experimental components. Their research results should not be confused with a validated authorship verdict from the application.

## Repository Structure

```text
K-Detect/
|-- app.py
|-- config/
|-- core/
|-- datasets/
|   |-- dataset documentation
|   |-- dataset processing scripts
|-- evaluation/
|   |-- evaluation scripts
|   |-- experiment logs
|   |-- selected result summaries
|-- models/
|   |-- traditional/
|   |-- ensemble/
|-- README.md
|-- MODEL_CARD.md
|-- PRIVACY.md
|-- RESPONSIBLE_USE.md
|-- LICENSE
|-- requirements.txt
```

Raw benchmark datasets, processed datasets, and trained model artifacts are not included in the public repository. Refer to the dataset documentation and experiment logs for information about data sources, preparation, and reproducibility.

## Technology Stack

- Python
- Streamlit
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Joblib
- Hugging Face Datasets
- Python-docx
- ReportLab

## Reproducibility

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment and install requirements:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The evaluation scripts and experiment logs document the research pipeline and experimental findings.

Some experiments require separately obtaining the corresponding datasets and regenerating trained model artifacts. Third-party dataset access and redistribution are subject to their respective terms and licensing conditions.

## Responsible Use

AI-text detection is inherently uncertain. Performance can vary with writing style, domain, language background, generation model, and subsequent human editing.

A classifier score is not proof of AI authorship or academic misconduct.

K-Detect must not be used as the sole basis for accusations, disciplinary decisions, or judgments about an individual's academic integrity.

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for further guidance.

## Privacy

The repository includes privacy documentation. Users should review it before using the software in any sensitive research or academic setting.

See [PRIVACY.md](PRIVACY.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).

## Project Status

K-Detect is an independent experimental research project. Its purpose is to investigate the strengths and limitations of existing AI-text detection approaches, not to claim a universally reliable detection solution.

Research findings, limitations, and methodological details are documented in:

- `evaluation/SCIHRA_EXPERIMENT_LOG.md`
- `evaluation/RAID_EXTERNAL_EXPERIMENT_LOG.md`

**Built to make AI-generated scientific writing detection research more transparent, reproducible, and responsible.**
