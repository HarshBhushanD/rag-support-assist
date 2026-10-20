
# Data Card: RAG Support Assistant

## 1. Dataset Overview

**Project:** RAG Support Assistant  
**Primary use:** Document-grounded question answering and support-ticket classification.

The project uses two sources:

1. **Technical documentation:** Official FastAPI documentation at https://fastapi.tiangolo.com/
2. **Support dataset:** Bitext Customer Support dataset hosted on Hugging Face at https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

## 2. Data Description

The technical documentation provides information about API development, request validation, routing, and error handling.

The Bitext dataset contains synthetic customer-support utterances and associated intent, category, and response information. We will inspect its actual schema before selecting fields for classification.

## 3. Intended Uses

- Answer questions using retrieved technical documentation.
- Evaluate document retrieval and answer quality.
- Train a baseline support-ticket classification model.
- Compare retrieval and classification experiments.

## 4. Out-of-Scope Uses

- Providing authoritative legal, medical, or financial advice.
- Treating synthetic examples as representative of all real customer interactions.
- Taking consequential actions without appropriate validation and approval.

## 5. Data Processing

Planned processing steps:

- Load and normalize supported document formats.
- Remove unnecessary markup and boilerplate.
- Detect and redact potential personally identifiable information (PII).
- Split documents into chunks while retaining source metadata.
- Track dataset versions and processing configurations.
- Keep raw and processed data separate.

## 6. Privacy and Security

The documentation and dataset will be inspected before processing. Potential PII will be detected using regular expressions and Presidio where appropriate. Automated redaction may miss sensitive information, so it will not be treated as a guarantee of complete anonymization.

No private customer records should be introduced into the project without authorization.

## 7. Licensing

The dataset's current Hugging Face card and repository license must be reviewed before redistribution or commercial use. The FastAPI documentation's license and reuse conditions must also be checked.

Dataset selection does not automatically grant permission to redistribute the source material.

## 8. Known Limitations

- Synthetic support examples may not reflect real-world language or class distributions.
- Technical documentation changes over time.
- Extraction, chunking, and retrieval may omit important context.
- Automated PII detection and LLM-generated answers can be incorrect.
- Evaluation results depend on the quality and coverage of the golden evaluation set.

## 9. Evaluation Plan

The project will measure:

- Retrieval hit rate and Mean Reciprocal Rank (MRR).
- Answer correctness and citation support.
- Classification precision, recall, F1 score, and confusion matrix.
- Latency and estimated cost per 1,000 queries.

Results will be recorded from actual runs; no performance claims are assumed in advance.

## 10. Versioning and Maintenance

Raw and processed data will be kept separate. DVC will be introduced for data versioning. Dataset updates will trigger appropriate re-ingestion and evaluation.
