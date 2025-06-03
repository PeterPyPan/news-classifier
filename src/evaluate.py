import argparse

import numpy as np
from datasets import load_dataset
from news_classifier import CLASS_LABELS
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate AG News Classifier')
    parser.add_argument(
        '--model_path',
        type=str,
        default='./model_output',
        help='Path to trained model directory',
    )
    args = parser.parse_args()

    # Load dataset
    print('Loading AG News test dataset...')
    dataset = load_dataset('ag_news')
    test_dataset = dataset['test']

    # Load tokenizer and model
    print(f'Loading model from {args.model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length', truncation=True, max_length=256
        )

    # Tokenize dataset
    print('Tokenizing test data...')
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    # Training arguments for evaluation
    eval_args = TrainingArguments(
        output_dir='./_tmp',
        per_device_eval_batch_size=64,
        do_train=False,
        do_predict=True,
        log_level='error',
    )

    # Initialize Trainer for evaluation
    trainer = Trainer(
        model=model,
        args=eval_args,
    )

    # Get predictions
    print('Running predictions on test set...')
    predictions = trainer.predict(tokenized_test)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Classification report
    print('Classification Report:')
    print(classification_report(labels, preds, target_names=CLASS_LABELS, digits=4))


if __name__ == '__main__':
    main()
