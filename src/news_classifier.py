from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

# Class labels mapping
CLASS_LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

MODEL_NAME = 'model_quantized.onnx'
QUANTIZED_MODEL_NAME = 'model.onnx'


class NewsClassifier:
    def __init__(self, models_path: Path, quantized=True):
        self.tokenizer = AutoTokenizer.from_pretrained(models_path)

        if quantized:
            onnx_path = models_path / QUANTIZED_MODEL_NAME
        else:
            onnx_path = models_path / MODEL_NAME
        if not onnx_path.is_file():
            raise FileNotFoundError(f'Onnx model {MODEL_NAME} found in: {models_path}')

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.log_severity_level = 3

        # Check for GPU availability
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=providers
        )

        # Print device info
        device = (
            'GPU' if 'CUDAExecutionProvider' in self.session.get_providers() else 'CPU'
        )
        print(f'Loaded ONNX model for inference on {device}')

    def predict(self, text: str):
        # Tokenize input
        inputs = self.tokenizer(
            text, return_tensors='np', padding=True, truncation=True, max_length=256
        )

        # Run inference
        logits = self.session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64),
            },
        )[0]

        # Process output
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)
        prediction = np.argmax(probabilities, axis=-1)[0]
        confidence = np.max(probabilities, axis=-1)[0]

        return {
            'class': int(prediction),
            'label': CLASS_LABELS[int(prediction)],
            'confidence': float(confidence),
        }
