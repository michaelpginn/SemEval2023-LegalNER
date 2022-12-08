# Load data from spaCy format
import spacy
from spacy.tokens import DocBin
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np

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


def create_model_and_trainer(train, dev, all_labels, tokenizer, pretrained='nlpaueb/legal-bert-base-uncased', batch_size, epochs):
    print("Creating model...")
    model = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=len(labels))
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
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
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
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    trainer = Trainer(
        model,
        args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return model, trainer
    


def main():
    train, labels = load_data()
    dev, _ = load_data('training/data/train.spacy')
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')
    train = process_dataset(train, tokenizer, labels)
    dev = process_dataset(dev, tokenizer, labels)
    model, trainer = create_model_and_trainer(train=train,
                                              dev=dev,
                                              all_labels=labels,
                                              tokenizer=tokenizer,
                                              batch_size=64,
                                              epochs=100)
    trainer.train()
    trainer.save_model('./output')

if __name__ == "__main__":
    main()