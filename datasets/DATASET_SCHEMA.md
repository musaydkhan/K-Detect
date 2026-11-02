

\# K-Detect Dataset Schema



\## Required fields



| Field | Description |

|---|---|

| text | Original text passage |

| label | 0 = Human, 1 = AI |

| source\_id | Unique source document identifier |

| domain | Academic discipline |

| generation\_model | AI model used, if applicable |

| editing\_status | Original, AI-generated, or human-edited AI |

| license | Dataset license or usage permission |

| split\_group | Identifier used to prevent data leakage |



\## Data principles



1\. Every passage must have a traceable source.

2\. Dataset licenses must permit our intended use.

3\. Human-written and AI-generated samples must be clearly distinguished.

4\. Human-edited AI text must be separately identified.

5\. Passages from the same original document must remain in the same data split.

6\. Test data must not leak into training or model selection.

7\. Sensitive personal information must not be collected unnecessarily.



\## Intended dataset categories



\- Human-written academic text

\- AI-generated academic text

\- Human-edited AI-generated text

\- Multiple academic disciplines

\- Different English proficiency levels



\## Important



Dataset labels are research annotations, not proof of authorship.
