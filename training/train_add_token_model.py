import spacy
from spacy.tokens import DocBin
from datasets import Dataset, load_metric, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import wandb
import sys
import train_sentence_classifier
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(spacy_file='data/train.spacy'):
    print("Loading data...")
    doc_bin = DocBin().from_disk(spacy_file)
    nlp = spacy.load('en_core_web_trf')
    docs = doc_bin.get_docs(nlp.vocab)
    
    all_sents = []
    all_labels = set()
    for doc in docs:
        new_sent = {'tokens': [token.text for token in doc],
                    'tags': [token.ent_iob_ + ("-" + token.ent_type_ if token.ent_type_ else '') for token in doc]}
        all_sents.append(new_sent)
        [all_labels.add(tag) for tag in new_sent['tags']]
    return Dataset.from_list(all_sents), sorted(list(all_labels))


def compute_class_preds(dataset, classifier_model: train_sentence_classifier.SentenceBinaryClassifier):
    """Adds a class label to each item in the dataset by running the predictive model"""
    print("Making class predictions")
    classifier_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

    def tokenize(row):
        return classifier_tokenizer(row['tokens'],
                                    truncation=True,
                                    is_split_into_words=True,
                                    padding='max_length',
                                    return_token_type_ids=False,
                                    return_tensors='pt')

    dataset_for_prediction = dataset.map(tokenize, batched=True)
    dataset_for_prediction.set_format(type="torch", columns=['input_ids', 'attention_mask'], device=device)

    class_labels = [False] * len(dataset_for_prediction)

    def predict(row, idx):
        preds = (classifier_model(row) > 0.5).tolist()
        for i, pred_i in zip(idx, range(len(preds))):
            class_labels[i] = preds[pred_i][0]
        return None

    dataset_for_prediction.map(predict, batched=True, batch_size=64, with_indices=True)
    return class_labels


def process_dataset(dataset, tokenizer, labels, classifier_model: train_sentence_classifier.SentenceBinaryClassifier):
    print("Processing dataset...")
    class_preds = compute_class_preds(dataset, classifier_model)

    def tokenize(row, idx):
        # Add special token for document type
        is_preamble = class_preds[idx]
        if is_preamble:
            row['tokens'].insert(0, '<PREAMBLE>')
        else:
            row['tokens'].insert(0, '<JUDGEMENT>')
        row['tags'].insert(0, 'O')

        tokenized = tokenizer(row['tokens'], truncation=True, is_split_into_words=True)
        aligned_labels = [-100 if i is None else labels.index(row['tags'][i]) for i in tokenized.word_ids()]
        tokenized['labels'] = aligned_labels

        return tokenized
    return dataset.map(tokenize, with_indices=True)


def load_test_data(tokenizer):
    all_rows = []
    for index, row in pd.read_json("../data/NER_TEST_DATA_FS.json").iterrows():
        all_rows.append({'text': row['data']['text'], 'meta': row['meta']['source']})

    test = Dataset.from_list(all_rows)

    def tokenize(row, idx):
        text = row['text']
        is_preamble = 'preamble' in row['meta']
        if is_preamble:
            text = '<PREAMBLE> ' + text
        else:
            text = ' <JUDGEMENT>' + text

        return tokenizer(text, truncation=True, is_split_into_words=False)
    return test.map(tokenize, with_indices=True)


metric = load_metric("seqeval")


def compute_metrics(pred, all_labels, verbose=False):
    predictions = pred[0]
    labels = pred[1]
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if verbose:
        return results
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def create_model_and_trainer(train, dev, all_labels, tokenizer, batch_size, epochs, run_name, pretrained='nlpaueb/legal-bert-base-uncased'):
    print("Creating model...")
    model = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=len(all_labels))
    model.resize_token_embeddings(len(tokenizer))
    args = TrainingArguments(
        f"checkpoints",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=3,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=3,
        report_to='wandb',
        run_name=run_name
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda pred: compute_metrics(pred, all_labels),
    )
    return model, trainer
    

def main():
    # Switch to nlpaueb/legal-bert-base-uncased
    final_train = False
    if sys.argv[1] == 'eval':
        eval_mode = True
    else:
        eval_mode = False
        if sys.argv[1] == 'final':
            final_train = True
            print("Training on combined train and dev data")
        wandb.init(project="legalner-custom", entity="seminal-2023-legalner")

    classifier_model = train_sentence_classifier.SentenceBinaryClassifier(hidden_size=128)
    classifier_model.load_state_dict(torch.load('./sentence-classification-model.pth'))
    classifier_model.to(device)

    train, labels = load_data()
    dev, _ = load_data('data/dev.spacy')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

    # For our custom tokens, let's add them
    tokenizer.add_tokens(['<PREAMBLE>', '<JUDGEMENT>'])

    train = process_dataset(train, tokenizer, labels, classifier_model=classifier_model)
    dev = dev.filter(lambda row: row['tags'][0] != '')
    dev = process_dataset(dev, tokenizer, labels, classifier_model=classifier_model)

    if final_train:
        train = concatenate_datasets([train, dev])
    model, trainer = create_model_and_trainer(train=train,
                                              dev=dev,
                                              all_labels=labels,
                                              tokenizer=tokenizer,
                                              batch_size=64,
                                              epochs=40,
                                              run_name='final_train',
                                              pretrained='./output' if eval_mode else 'roberta-base')


    if not eval_mode:
        trainer.train()
        trainer.save_model('./output')

    # Evaluate regardless
    predictions = trainer.predict(dev)
    all_metrics = compute_metrics(predictions, all_labels=labels, verbose=True)
    for key in all_metrics:
        print(key, ':\t', all_metrics[key])

if __name__ == "__main__":
    main()