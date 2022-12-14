import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List
import random 
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tokenizer

class BatchTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        
    def __call__(self, sentence: List[str]):
        return self.tokenizer(
            sentence,
            padding='max_length', 
            truncation=True,
            return_token_type_ids=False,
            return_tensors='pt'
        ).to(device)


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]
        
# Batch labels as well
def encode_labels(labels: List[int]) -> torch.FloatTensor:
    return torch.FloatTensor([int(l) for l in labels]).to(device)

# Declare model

class SentenceBinaryClassifier(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.legal_bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.bert_hidden_dim = self.legal_bert.config.hidden_size
        for param in self.legal_bert.parameters():
            param.requires_grad = False
        
        self.hidden_layer = torch.nn.Linear(self.bert_hidden_dim, self.hidden_size)
        self.relu = torch.nn.ReLU()
        
        self.classifier = torch.nn.Linear(self.hidden_size, 1)
        
    def forward(self, sentences) -> torch.Tensor:
        src = self.legal_bert(**sentences).pooler_output
        src = self.relu(self.hidden_layer(src))
        out = self.classifier(src)
        return torch.sigmoid(out)
    
def predict(model, sents):
    return (model(sents) > 0.5).float()


def eval(labels, preds):
    print(f"F1 Score: {f1_score(labels, preds, average='macro')}")
    print(f"Prec: {precision_score(labels, preds, average='macro')}")
    print(f"Rec: {recall_score(labels, preds, average='macro')}")


def training_loop(num_epochs, train_sentences, train_labels, dev_sentences, dev_labels, optimizer, model):
    print("Training...")
    loss_func = torch.nn.BCELoss()
    batches = list(zip(train_sentences, train_labels))
    random.shuffle(batches)
    dev_batches = list(zip(dev_sentences, dev_labels))
    random.shuffle(dev_batches)
    
    for i in range(num_epochs):
        losses = []
        for sents, labels in tqdm(batches):
            optimizer.zero_grad()
            preds = model(sents).squeeze(1)
            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"epoch {i}, loss: {sum(losses)/len(losses)}")
        print("Evaluating dev")
        dev_preds = []
        dev_labels = []
        for sents, labels in tqdm(dev_batches):
            pred = predict(model, sents).squeeze(1)
            dev_preds.extend(pred.cpu())
            dev_labels.extend(list(labels.cpu().numpy()))
        eval(dev_labels, dev_preds)
    return model

def main():
    if sys.argv[1] == 'eval':
        eval_mode = True
    else:
        eval_mode = False

    tokenizer = BatchTokenizer()
    batch_size = 64

    if not eval_mode:
        print("Loading training data")
        # Load training data
        preamble = pd.read_json("../data/NER_TRAIN/NER_TRAIN_PREAMBLE.json")
        preamble_texts = [item['text'] for item in preamble['data']]
        judgement = pd.read_json("../data/NER_TRAIN/NER_TRAIN_JUDGEMENT.json")
        judgement_texts = [item['text'] for item in judgement['data']]

        # Create labels for each of the sentences
        all_texts = preamble_texts + judgement_texts
        all_labels = [1] * len(preamble_texts) + [0] * len(judgement_texts)
    
        # Create batches
        train_input_batches = [tokenizer(b) for b in chunk(all_texts, batch_size)]
        train_label_batches = [encode_labels(b) for b in chunk(all_labels, batch_size)]

    # Process dev data as well
    print("Processing dev data")
    preamble_texts_dev = [item['text'] for item in pd.read_json("../data/NER_DEV/NER_DEV_PREAMBLE.json")['data']]
    judgement_texts_dev = [item['text'] for item in pd.read_json("../data/NER_DEV/NER_DEV_JUDGEMENT.json")['data']]
    all_texts_dev = preamble_texts_dev + judgement_texts_dev
    all_labels_dev = [1] * len(preamble_texts_dev) + [0] * len(judgement_texts_dev)
    dev_sents_batches = [tokenizer(b) for b in chunk(all_texts_dev, batch_size)]
    dev_labels_batches = [encode_labels(b) for b in chunk(all_labels_dev, batch_size)]

    if not eval_mode:
        model = SentenceBinaryClassifier(hidden_size=128).to(device)
        training_loop(
            num_epochs=10,
            train_sentences=train_input_batches,
            train_labels=train_label_batches,
            dev_sentences=dev_sents_batches,
            dev_labels=dev_labels_batches,
            optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
            model=model
        )
        torch.save(model.state_dict(), "./sentence-classification-model.pth")
    else:
        # Load the trained model and evaluate
        trained_model = SentenceBinaryClassifier(hidden_size=128).to(device)
        trained_model.load_state_dict(torch.load('./sentence-classification-model.pth'))
        dev_batches = list(zip(dev_sents_batches, dev_labels_batches))
        dev_preds = []
        dev_labels = []
        for sents, labels in tqdm(dev_batches):
            pred = predict(trained_model, sents).squeeze(1)
            dev_preds.extend(pred.cpu())
            dev_labels.extend(list(labels.cpu().numpy()))
        eval(dev_labels, dev_preds)

if __name__ == "__main__":
    main()