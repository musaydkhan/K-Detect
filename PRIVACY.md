# K-Detect Privacy Policy

**Last updated:** November 2026

## 1. Our Commitment

Academic writing may contain personal, unpublished, or sensitive information. K-Detect aims to minimize unnecessary collection and storage of submitted documents.

This policy describes the current application's document-processing behavior. It does not guarantee the privacy practices of third-party hosting providers or deployment environments.

## 2. Current Application Behavior

The current application supports TXT and DOCX uploads, as well as directly pasted text.

Based on the current source code:

- TXT and DOCX files are read for text extraction and analysis.

- Extracted text and analysis results are held in the application's Streamlit session state.

- PDF reports, annotated DOCX files, and CSV exports are generated for user download.

- Report and document generation uses in-memory objects in the application code.

- The application does not intentionally send submitted documents to an external AI service.

The source code does not establish that every hosting environment processes or retains data exclusively on the user's device.

## 3. Document Storage and Retention

The current document-analysis workflow does not intentionally save uploaded documents as persistent files.

However:

- Uploaded content and analysis results may remain temporarily available in application memory or session state.

- Runtime behavior and session retention may depend on the Streamlit deployment environment.

- Hosting providers may maintain their own operational logs or infrastructure records.

K-Detect does not claim that all temporary data is immediately erased when a session ends.

## 4. Third-Party Services and Hosting

The current application source does not show an intentional integration with an external AI API for document analysis.

Users should distinguish between:

- **Local execution:** Running the application on their own computer.

- **Hosted execution:** Accessing an application deployed on a remote server or hosting platform.

When using a hosted deployment, uploaded documents are processed by that deployment's server. Its hosting provider may have separate data-handling practices.

Users should review the relevant hosting provider's privacy policy before submitting confidential or sensitive documents.

## 5. Research Data

K-Detect's research experiments use external datasets to evaluate text-classification approaches.

These research datasets are separate from documents submitted by application users. User-submitted documents are not intentionally added to research datasets or used to train models.

## 6. User Recommendations

Users should:

- Avoid uploading confidential, personally identifiable, or unpublished material to deployments they do not trust.

- Run the application locally when they require greater control over document handling.

- Review the source code and deployment configuration before using K-Detect with sensitive documents.

## 7. Open-Source Transparency

K-Detect's source code is publicly available for inspection.

We welcome independent review of its document-processing, privacy, and security practices.

Open-source availability does not constitute a formal security audit or guarantee the absence of vulnerabilities.

## 8. Future Changes

Future versions may introduce additional functionality, integrations, or storage mechanisms.

Any significant changes to document processing, data collection, storage, or external services should be reflected in this policy.

## 9. Contact

For questions, suggestions, or privacy concerns, please open an issue in the K-Detect GitHub repository.

---

**K-Detect - Your writing deserves transparency. Your data deserves privacy.**

## Update, November 2026
- No accounts. Text is processed in memory and is not stored by K-Detect.
- The similarity check sends short phrases (about nine words each) from the text to OpenAlex, Europe PMC, arXiv and, if enabled, Brave Search.
  Those services have their own privacy policies.
