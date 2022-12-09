# SemEval2023-LegalNER

https://sites.google.com/view/legaleval/home

This repo is associated with the subtask "Sub-task B: Legal Named Entities Extraction (L-NER)" of SemEval 2023, Task 6.

### Trained Models
https://drive.google.com/drive/folders/1YmcGInc6R4_qUGzbPtHVuQ8eQut6GOGn?usp=sharing

## Task
The goal of this project is to perform Named Entity Recognition (NER) on Indian legal texts. General-purpose NER systems tend to perform poorly on legal texts, so a specialized system is desirable.

## Experiments

We have noticed that whether certain entity types are tagged is dependent on whether the sentence comes from the 

1. **Baseline and Modifications.** The baseline uses the spaCy transition-based NER parser. We will try this (and slight modifications).
    1. **Baseline system [[2]](#2)** - (https://github.com/Legal-NLP-EkStep/legal_NER) Uses RoBERTa as the embeddings and spaCy Transition-based NER parser.
    2. **LegalBERT** - Replace the RoBERTa with [LegalBERT](#1).
    3. **Higher Dropout** - The baseline dropout is very low. Try increasing it to a standard value, such as 0.3.
    4. **Larger Hidden State** - Increase the hidden state of the NER model to 128.
2. **Custom transformer-based model.** spaCy's parser is powerful, but somewhat opaque and hard to modify. We will create a custom system, using PyTorch/Huggingface transformer models with a token classification head on top. Our primary experiment is how we can add information about the document type, since it affects how entities should be tagged. 
    1. **LegalBERT, no document info** - Train a baseline LegalBERT with a token classification head on top.
    2. **RoBERTa, no document info** - Same as previous
    3. **Add document type token** - Append a special token to each document that indicates the document type.
    4. **Pretrain transformer for classification** - Pretrain first for classification, then decapitate and finetune for NER.
    5. **Add document type to FFN** - Append a node to the FFN at the head of the model that indicates the document type.
    6. 
  
6. **Combined System** - As in [Jiang et al](#3), leverage the output of the spaCy and Stanford NER models.

## Experimental Results

> For the spaCy models, train with `python3 -m spacy train config.cfg --output ./output --gpu-id 0`.
> For the custom models, train with `python3 train_custom_model.py`.

| Experimental Conditions | Trained on Machine # | F1 | Prec | Rec |
| --- | --- | --- | --- | --- |
| 1.i. Baseline | 1 | 0.903 | 0.903 | 0.904 |
| 1.ii. LegalBERT as embeddings | 1 | 0.893 | 0.892 | 0.894 |
| 1.iii. Higher Dropout (0.3) | 1 | 0.901 | 0.901 | 0.901 |
| 1.iv. Larger Hidden State | 1 | 0.890 | 0.891 | 0.889 |

| Model | F1 | Prec | Rec |
| Sentence Classifier | | | |


### Machines
1. Google Cloud Compute VM, n1-standard-4, 1 x NVIDIA T4, Debian


## References
<a id="1">[1]</a>  Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The Muppets straight out of Law School. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 2898–2904, Online. Association for Computational Linguistics.

<a id="2">[2]</a> Prathamesh Kalamkar, Astha Agarwal, Aman Tiwari, Smita Gupta, Saurabh Karn, and Vivek Raghavan. 2022. Named Entity Recognition in Indian court judgments. arXiv:2211.03442 [cs].

<a id="3">[3]</a> Ridong Jiang, Rafael E. Banchs, and Haizhou Li. 2016. Evaluating and Combining Name Entity Recognition Systems. In Proceedings of the Sixth Named Entity Workshop, pages 21–27, Berlin, Germany. Association for Computational Linguistics.

