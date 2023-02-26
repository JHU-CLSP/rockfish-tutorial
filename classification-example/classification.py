import torch, codecs, random
from torch.utils.data import DataLoader
# from datasets import load_metric
# from google.colab import output
# import matplotlib.pyplot as plt
# import numpy as np
# output.enable_custom_widget_manager()
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate as evaluate
# from datasets import evaluate
from transformers import get_scheduler
# from transformers import DistilBertForSequenceClassification
from transformers import AutoModelForSequenceClassification
import argparse
import subprocess

def print_gpu_memory():
    """ Print the amount of GPU memory used """
    # check if gpu is available
    if torch.cuda.is_available():
        print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
        print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
        print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

        p = subprocess.check_output('nvidia-smi')
        print(p.decode("utf-8"))


class BoolQADataset(torch.utils.data.Dataset):
    """ Dataset for the IMDB dataset """

    def __init__(self, passages, questions, answers, tokenizer, max_len):
        self.passages = passages
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        passage = str(self.passages[index])
        question = self.questions[index]
        answer = self.answers[index]

        input = question + " [SEP] " + passage
        # print(f"Input: {input} - Output: {answer}")
        encoded_review = self.tokenizer.encode_plus(
            input,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': encoded_review['input_ids'][0],
            'attention_mask': encoded_review['attention_mask'][0],
            'labels': torch.tensor(answer, dtype=torch.long)
        }

def update_metrics(metrics, predictions, labels):
    """ Update a list of metrics with new predictions and labels

    :param list<datasets.Metric> metrics: list of metrics
    :param torch.Tensor predictions: tensor of predictions of shape (1, batch_size)
    :param torch.Tensor labels: tensor of labels of shape (1, batch_size)

    :return None
    """
    # Nothing TODO here! This updates metrics based on a batch of predictions
    # and a batch of labels.
    for metric in metrics:
        # references = torch.zeros(predictions.shape, dtype=torch.long)
        # print(references)
        # references[:, 0] = 1-labels
        # references[:, 1] = labels

        metric.add_batch(predictions=predictions, references=labels)


def evaluateModel(model, test_dataloader, device, metric_strs):
    """ Evaluate a PyTorch Model

    :param torch.nn.Module model: the model to be evaluated
    :param torch.utils.data.DataLoader test_dataloader: DataLoader containing testing examples
    :param torch.device device: the device that we'll be training on
    :param list<string> metric_strs: the names of Hugging Face metrics to use

    :return dictionary<string, float>: dictionary of metrics names mapped to their values
    """
    # load metrics
    metrics = [evaluate.load(x) for x in metric_strs]  # could add more here!
    model.eval()

    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        # print(output)
        predictions = output.logits
        # print(predictions)
        # Update the metrics
        # update_metrics(...)
        predictions = torch.argmax(predictions, dim=1)
        update_metrics(metrics, predictions, batch['labels'])

    # compute and return metrics
    computed = {}
    for m in metrics:
        computed = {**computed, **m.compute()}

    return computed


def train(mymodel, num_epochs, train_dataloader, validation_dataloader, device, lr):
    """ Train a PyTorch Module

    :param torch.nn.Module mymodel: the model to be trained
    :param int num_epochs: number of epochs to train for
    :param torch.utils.data.DataLoader train_dataloader: DataLoader containing training examples
    :param torch.utils.data.DataLoader validation_dataloader: DataLoader containing validation examples
    :param torch.device device: the device that we'll be training on

    :return None
    """

    # here, we use the AdamW optimzer. Use torch.optim.Adam.
    # intantiate it on the untrained model parameters with a learning rate of 5e-5
    # optimizer = ...
    optimizer = torch.optim.Adam(
        mymodel.parameters(),
        lr=lr
    )

    # now, we set up the learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=50,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    loss = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print_gpu_memory()

        # put the model in training mode (important that this is done each epoch,
        # since we put the model into eval mode during validation)
        mymodel.train()

        # load metrics
        # metrics = [evaluate.load(x) for x in ["accuracy"]]  # could add more here!
        train_accuracy = evaluate.load('accuracy')

        print(f"Epoch {epoch + 1} training:")

        for i, batch in enumerate(train_dataloader):
            # TODO: fill in!
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output = mymodel(input_ids=input_ids, attention_mask=attention_mask)
            # print(output)
            predictions = output.logits
            # predictions = torch.argmax(predictions, dim=1)
            output = loss(predictions, batch['labels'].to(device))
            # print(output)
            output.backward()

            # update the optimizer # NEW
            optimizer.step()
            lr_scheduler.step()

            predictions = torch.argmax(predictions, dim=1)

            # update metrics
            # update_metrics(...)
            # update_metrics(metrics, predictions, batch['labels'])
            train_accuracy.add_batch(predictions=predictions, references=batch['labels'])

            if i % 50 == 0:
                print(f" ===> Epoch {epoch + 1}, batch {i}; average training metrics: accuracy={train_accuracy.compute()}")

            # if i % 500 == 0:
            #     # print the epoch's average metrics
            #     print(
            #         f"Epoch {epoch + 1}, after batch {i} ; average training metrics: accuracy={metrics[0].compute()['accuracy']}")
            #
            #     # normally, validation would be more useful when training for many epochs
            #     print("Running validation:")
            #     # TODO: evaluate mymodel on validation dataset
            #     # val_metrics = evaluate(...)
            #
            #     val_metrics = evaluate(mymodel, validation_dataloader, device, ['accuracy'])['accuracy']
            #     # baseline_result = evaluate(untrained_model, test_dataloader, device, ['accuracy'])['accuracy']
            #
            #     print(f"Epoch {epoch + 1} validation: accuracy={val_metrics}")


def run_experiment(num_epochs, lr, batch_size, device):
    # download dataset
    dataset = load_dataset("boolq")
    dataset = dataset.shuffle()  # shuffle the data

    # use this for the tokenizer argument of the TweetDataset
    mytokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # since the dataset does not come with any validation data,
    # split the training data into "train" and "dev"
    dataset_train_subset = dataset['train'][:8000]
    dataset_dev_subset = dataset['validation']

    # use the subset of the test set for making the experimentation faster
    dataset_test_subset = dataset['train'][8000:]
    max_len = 128

    train_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    validation_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )
    test_dataset = BoolQADataset(
        passages=list(dataset_train_subset['passage']),
        questions=list(dataset_train_subset['question']),
        answers=list(dataset_train_subset['answer']),
        tokenizer=mytokenizer,
        max_len=max_len
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # TODO: load the distilbert-base-uncased pre-trained model, use DistilBertForSequenceClassification
    # from Hugging Face (transformers), read their documentation to do this.
    pretrained_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    print("Movidng model to device ..." + str(device))
    pretrained_model.to(device)

    # TODO: train!
    # train(...)
    # train(pretrained_model, num_epochs, train_dataloader, validation_dataloader, device, lr=5e-5)
    # train(pretrained_model, num_epochs, train_dataloader, validation_dataloader, device, lr=1e-4)
    # train(pretrained_model, num_epochs, train_dataloader, validation_dataloader, device, lr=5e-4)
    train(pretrained_model, num_epochs, train_dataloader, validation_dataloader, device, lr=lr)

    print_gpu_memory()

    """You should've seen a sneak-peak of the model's performance based on the validation accuracies!

    ### 3. Evaluating the fine-tuned model:
    """

    # TODO: Evaluate the fine-tuned model on the test dataset
    # finetuned_result = evaluate(...)['accuracy']
    # pretrained_result = evaluate(pretrained_model, test_dataloader, device, metrics)['accuracy']
    # print(pretrained_result)


# create the __main__ function
if __name__ == "__main__":
    # get use arguments for num_epochs, lr, batch_size, device
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # run the experiment
    run_experiment(args.num_epochs, args.lr, args.batch_size, args.device)


