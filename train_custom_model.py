import spacy
from spacy.tokens import DocBin
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import wandb
import sys
import json

def load_data(spacy_file='training/data/train.spacy'):
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


def process_dataset(dataset, tokenizer, labels):
    print("Processing dataset...")
    def tokenize(row):
        tokenized = tokenizer(row['tokens'], truncation=True, is_split_into_words=True)
        aligned_labels = [-100 if i is None else labels.index(row['tags'][i]) for i in tokenized.word_ids()]
        tokenized['labels'] = aligned_labels
        return tokenized
    return dataset.map(tokenize)


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
    if sys.argv[1] == 'eval':
        eval_mode = True
    else:
        eval_mode = False
        wandb.init(project="legalner-custom", entity="seminal-2023-legalner")
    train, labels = load_data()
    dev, _ = load_data('training/data/dev.spacy')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
    train = process_dataset(train, tokenizer, labels)
    dev = dev.filter(lambda row: row['tags'][0] != '')
    dev = process_dataset(dev, tokenizer, labels)
    model, trainer = create_model_and_trainer(train=train,
                                              dev=dev,
                                              all_labels=labels,
                                              tokenizer=tokenizer,
                                              batch_size=64,
                                              epochs=75,
                                              run_name='roberta-baseline',
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