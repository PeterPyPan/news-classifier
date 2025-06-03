import argparse
from datetime import datetime
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

DATA_DIR = Path(__file__).parents[1] / 'data'
OUTPUT_DIR = DATA_DIR / 'output'
MODEL_NAME = 'distilbert-base-uncased'

# Make sure we are running on GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'We are training on {DEVICE}')


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train DistilBERT on AG News')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument(
        '--learning_rate', type=float, default=2e-5, help='Learning rate'
    )
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')

    args = parser.parse_args()

    # extract hyperparameters
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    n_train_epochs = args.epochs
    weight_decay = args.weight_decay

    # Set seed for reproducibility
    set_seed(42)

    # Load dataset
    print('Loading AG News dataset...')
    dataset = load_dataset('ag_news')

    # Load accuracy metric
    accuracy_metric = evaluate.load('accuracy')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length', truncation=True, max_length=256
        )

    # Tokenize dataset
    print('Tokenizing data...')
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Model initialization
    print('Initializing model...')
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=4)

    # Compute metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Training arguments
    output_dir = OUTPUT_DIR / datetime.now().strftime('%Y%m%d%H%M%S')
    logs_dir = output_dir / 'logs'
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        logging_dir=str(logs_dir),
        report_to='tensorboard',
        fp16=True,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        compute_metrics=compute_metrics,
    )

    # Start training
    print('Training started...')
    trainer.train()

    # Save final model
    print(f'Saving model to {output_dir}...')
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate on test set
    results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Final test accuracy: {results['eval_accuracy']:.4f}")


if __name__ == '__main__':
    main()
