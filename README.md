# SemEval2023-LegalNER

https://sites.google.com/view/legaleval/home

This repo is associated with the subtask "Sub-task B: Legal Named Entities Extraction (L-NER)" of SemEval 2023, Task 6.

## Task
The goal of this project is to perform Named Entity Recognition (NER) on Indian legal texts. General-purpose NER systems tend to perform poorly on legal texts, so a specialized system is desirable.

## Experiments

1. **Baseline system [[2]](#2)** - (https://github.com/Legal-NLP-EkStep/legal_NER) Uses RoBERTa as the embeddings and spaCy Transition-based NER parser.
2. **LegalBERT** - Replace the RoBERTa with [LegalBERT](#1).
3. **Higher Dropout** - The baseline dropout is very low. Try increasing it to a standard value, such as 0.3.
4. **Combined System** - As in [Jiang et al](#3), leverage the output of the spaCy and Stanford NER models.

## Experimental Results

Train with `python -m spacy train config.cfg --output ./output --gpu-id 0`.

| Experimental Conditions | Trained on Machine # | F1 | Prec | Acc |
| --- | --- | --- | --- | --- |
| 1. Baseline | 1 | | |
| 2. LegalBERT as embeddings | | | |
| 3. Higher Dropout (0.3) | | | |
| 3. Combined System | | | |


### Machines
1. Google Cloud Compute VM, n1-standard-4, 1 x NVIDIA T4, Debian

### To be continued...

## References
<a id="1">[1]</a>  Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The Muppets straight out of Law School. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2898–2904, Online. Association for Computational Linguistics.

<a id="2">[2]</a> Prathamesh Kalamkar, Astha Agarwal, Aman Tiwari, Smita Gupta, Saurabh Karn, and Vivek Raghavan. 2022. Named Entity Recognition in Indian court judgments. arXiv:2211.03442 [cs].

<a id="3">[3]</a> Ridong Jiang, Rafael E. Banchs, and Haizhou Li. 2016. Evaluating and Combining Name Entity Recognition Systems. In Proceedings of the Sixth Named Entity Workshop, pages 21–27, Berlin, Germany. Association for Computational Linguistics.

