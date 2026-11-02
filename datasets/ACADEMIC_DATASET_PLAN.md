

\# K-Detect Academic Dataset Plan



\## Primary candidate



Name: SciHRA-Detect

Source: https://github.com/mithuGit/SciHRA-Detect



Reported size: 3,000 scientific papers

Domains: 10

Text categories:

\- HGT: Human-generated

\- ART: AI-revised

\- AGT: AI-generated



Generation model: GPT-4o



Potential use:

\- Academic domain generalization

\- Human vs AI classification

\- Human vs AI-revised classification

\- Three-class authorship experiments



License status:

\- Repository lists Apache-2.0.

\- Underlying source-paper permissions require verification.

\- Do not redistribute raw data until cleared.



\## Secondary candidate



Name: AIGTxt

Source: https://data.mendeley.com/datasets/y9bj7734vf/1



Reported size: 10,821 samples

Categories:

\- Human-written

\- AI-generated

\- Mixed



License: CC BY-NC 4.0



Potential use:

\- Independent academic evaluation

\- Mixed-authorship analysis



\## Experimental principles



1\. Preserve original source identifiers.

2\. Group all related samples by original paper.

3\. Never allow related versions into different splits.

4\. Keep an untouched external test dataset.

5\. Record generation model and source metadata.

6\. Evaluate AI-revised text separately.

7\. Do not interpret detector predictions as proof of authorship.

8\. Document all dataset licenses and restrictions.



\## Planned experiments



1\. Evaluate existing HC3 models on academic text without retraining.

2\. Train traditional models on academic training data.

3\. Evaluate on held-out academic papers.

4\. Compare performance across HGT, ART and AGT.

5\. Investigate domain-specific false positives.
