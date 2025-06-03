from pathlib import Path

from datasets import load_dataset
from news_classifier import NewsClassifier
from sklearn.metrics import classification_report
from tqdm import tqdm

CLASS_LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

MODELS_DIR = Path(__file__).parents[1] / 'models'


def evaluate_model(classifier, test_data):
    # Get predictions
    predictions = []
    for example in tqdm(test_data):
        result = classifier.predict(example['text'])
        predictions.append(result['class'])

    # Get true labels
    true_labels = test_data['label']

    # Classification report
    print('ONNX Model Performance:')
    print(
        classification_report(
            true_labels, predictions, target_names=CLASS_LABELS, digits=4
        )
    )


def main():
    # Load dataset
    dataset = load_dataset('ag_news')
    test_data = dataset['test']

    # classifier
    classifier = NewsClassifier(models_path=MODELS_DIR, quantized=False)
    print('## Unquantized ONNX model')
    evaluate_model(classifier=classifier, test_data=test_data)

    # quantized classifier
    classifier_quant = NewsClassifier(models_path=MODELS_DIR, quantized=True)
    print('## Quantized ONNX model')
    evaluate_model(classifier=classifier_quant, test_data=test_data)


if __name__ == '__main__':
    main()
