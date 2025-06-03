from pathlib import Path

from news_classifier import NewsClassifier

# Define the model directory path
MODELS_DIR = Path(__file__).parents[1] / 'models'


def predict(text, quantized=True):
    # Initialize the classifier
    classifier = NewsClassifier(models_path=MODELS_DIR, quantized=quantized)

    # Run prediction
    return classifier.predict(text)


if __name__ == '__main__':
    samples = [
        'Stock markets reach all-time high amid economic recovery',
        'NASA discovers Earth-like exoplanet in habitable zone',
        'Max Verstappen won the formula 1 race in Bahrein',
        'Frail Pope Celebrates Mass at Lourdes',
    ]

    for text in samples:
        result = predict(text=text, quantized=True)
        print(f'Input text: {text}')
        print(f" - Prediction: {result['label']} (Class {result['class']})")
        print(f" - Confidence: {result['confidence']:.4f}\n")
