# SemEval2023-LegalNER

https://sites.google.com/view/legaleval/home

This repo is associated with the subtask "Sub-task B: Legal Named Entities Extraction (L-NER)" of SemEval 2023, Task 6.

## Task
The goal of this project is to perform Named Entity Recognition (NER) on Indian legal texts. General-purpose NER systems tend to perform poorly on legal texts, so a specialized system is desirable.

## Experimental Results

Train with `python -m spacy train config.cfg --output ./output --gpu-id 0`.

| Experimental Conditions | Trained on | F1 | Prec | Acc |
| --- | --- | --- | --- | --- |
| Baseline (https://github.com/Legal-NLP-EkStep/legal_NER) | 1 | | |
| LegalBERT as embeddings | | | |


### Machines
1. Google Cloud Compute VM, n1-standard-4, 1 x NVIDIA T4, Debian

### To be continued...
