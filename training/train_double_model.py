import spacy
from spacy.tokens import DocBin
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
import numpy as np
import wandb
import sys
import train_sentence_classifier
import torch

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
    # return [0] * len(dataset)
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
        tokenized = tokenizer(row['tokens'], truncation=True, is_split_into_words=True)
        aligned_labels = [-100 if i is None else labels.index(row['tags'][i]) for i in tokenized.word_ids()]
        tokenized['labels'] = aligned_labels

        """Store the class for the row"""
        tokenized['doc_class'] = class_preds[idx]

        return tokenized
    return dataset.map(tokenize, with_indices=True)


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


class DoubleTokenClassifierModel(torch.nn.Module):
    def __init__(self, all_labels, pretrained):
        super().__init__()
        self.preamble_model = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=len(all_labels))
        self.judgement_model = AutoModelForTokenClassification.from_pretrained(pretrained, num_labels=len(all_labels))

    def forward(self, doc_class: torch.LongTensor, input_ids: torch.LongTensor = None, attention_mask = None, position_ids = None, labels = None):
        row_mask = doc_class > 0
        flipped_row_mask = ~row_mask

        preamble_batch_input_ids = None if input_ids is None else input_ids[row_mask]
        preamble_attention_mask = None if attention_mask is None else attention_mask[row_mask]
        preamble_position_ids = None if position_ids is None else position_ids[row_mask]
        preamble_labels = None if labels is None else labels[row_mask]
        preamble_output = self.preamble_model(input_ids=preamble_batch_input_ids, attention_mask=preamble_attention_mask, position_ids=preamble_position_ids, labels=preamble_labels)

        judgement_batch_input_ids = None if input_ids is None else input_ids[flipped_row_mask]
        judgement_attention_mask = None if attention_mask is None else attention_mask[flipped_row_mask]
        judgement_position_ids = None if position_ids is None else position_ids[flipped_row_mask]
        judgement_labels = None if labels is None else labels[flipped_row_mask]
        judgement_output = self.judgement_model(input_ids=judgement_batch_input_ids,
                                                attention_mask=judgement_attention_mask,
                                                position_ids=judgement_position_ids, labels=judgement_labels)

        loss = (preamble_output.loss or 0) + (judgement_output.loss or 0)

        # Pick the right logits together
        preamble_batch_index = 0
        judgement_batch_index = 0
        logits = []
        for i in range(len(doc_class)):
            if row_mask[i]:
                logits.append(preamble_output.logits[preamble_batch_index])
                preamble_batch_index += 1
            else:
                logits.append(judgement_output.logits[judgement_batch_index])
                judgement_batch_index += 1

        return TokenClassifierOutput(loss=loss, logits=torch.stack(logits).to(device))



def create_model_and_trainer(train, dev, all_labels, tokenizer, batch_size, epochs, run_name, pretrained='nlpaueb/legal-bert-base-uncased'):
    print("Creating model...")
    model = DoubleTokenClassifierModel(all_labels, pretrained)
    args = TrainingArguments(
        f"checkpoints",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=6,
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

    classifier_model = train_sentence_classifier.SentenceBinaryClassifier(hidden_size=128)
    classifier_model.load_state_dict(torch.load('./sentence-classification-model.pth'))
    classifier_model.to(device)

    train, labels = load_data()
    dev, _ = load_data('data/dev.spacy')

    tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True)

    train = process_dataset(train, tokenizer, labels, classifier_model=classifier_model)
    dev = dev.filter(lambda row: row['tags'][0] != '')
    dev = process_dataset(dev, tokenizer, labels, classifier_model=classifier_model)

    model, trainer = create_model_and_trainer(train=train,
                                              dev=dev,
                                              all_labels=labels,
                                              tokenizer=tokenizer,
                                              batch_size=32,
                                              epochs=40,
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