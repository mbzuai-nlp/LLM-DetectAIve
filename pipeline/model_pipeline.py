import torch
import optuna
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)

from dataset import prepare_dataset, save_tokenized_dataset

'''
tokenization functions
'''
def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512)

def tokenize_and_prepare_dataset(dataset, tokenizer):
    tokenized_datasets = dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["text", "domain"])
    tokenized_datasets.set_format("torch")
    return tokenized_datasets

'''
training & hyperparamters optimization functions
'''
def get_model_and_tokenizer(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model, tokenizer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def objective(trial, model, tokenized_datasets, max_epochs, metric='eval_f1'):
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-1, log=True)
    epoch = trial.suggest_int("epoch", 1, max_epochs)

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=epoch,
        weight_decay=weight_decay,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results[metric]

def train_model(model_name, tokenized_datasets, num_labels, num_trials=5, max_epochs=5):
    model, _ = get_model_and_tokenizer(model_name, num_labels)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, model, tokenized_datasets, max_epochs), n_trials=num_trials)

    print(f"Best hyperparameters for {model_name}:", study.best_params)
    print(f"Best F1 score for {model_name}:", study.best_value)

    # Train with best hyperparameters
    best_training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        num_train_epochs = study.best_params["epoch"],
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        learning_rate=study.best_params["learning_rate"],
        weight_decay=study.best_params["weight_decay"],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        load_best_model_at_end=True,
        eval_steps = 500,
        logging_steps = 500
    )

    trainer = Trainer(
        model=model,
        args=best_training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer, model



'''
Evaluation
'''

def evaluate_model(trainer, tokenized_datasets):
    results = trainer.evaluate(tokenized_datasets['test'])
    print("Test set results:", results)

    predictions = trainer.predict(tokenized_datasets['test'])
    preds = torch.argmax(torch.tensor(predictions.predictions), axis=-1).cpu().numpy()
    true_labels = tokenized_datasets['test']['labels'].numpy()

    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

'''
model and tokenizer save and load
'''
def save_model_and_tokenizer(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def load_model_and_tokenizer(model_name, load_path, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(load_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    return model, tokenizer


'''
Integrated pipeling
'''
def run_training_pipeline(file_path, labels_dict, model_name, num_labels, sample_frac=1.0, num_trials=5, num_epochs=5, save_dir = "."):
    # Prepare dataset
    dataset = prepare_dataset(file_path, labels_dict, sample_frac=sample_frac)

    print(f"Training {model_name}...")

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_name, num_labels)

    # Tokenize dataset
    tokenized_datasets = tokenize_and_prepare_dataset(dataset, tokenizer)

    # Save tokenized dataset
    save_tokenized_dataset(tokenized_datasets, f"{save_dir}/tokenized_{model_name}")

    # Train model
    trainer, trained_model = train_model(model_name, tokenized_datasets, num_labels, num_trials, num_epochs)

    # Evaluate model
    evaluate_model(trainer, tokenized_datasets)

    # Save model and tokenizer
    save_model_and_tokenizer(trained_model, tokenizer, f"{save_dir}/fine_tuned_{model_name}")
