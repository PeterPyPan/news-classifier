import argparse
from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODELS_DIR = Path(__file__).parents[1] / 'models'


def export_onnx(model_path):
    MODELS_DIR.mkdir(exist_ok=True)

    # Load model and tokenizer
    print(f'Loading model from {model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Create dummy input for tracing
    print('Creating sample input for tracing...')
    dummy_text = 'This is a sample input for model tracing'
    inputs = tokenizer(
        dummy_text, return_tensors='pt', padding=True, truncation=True, max_length=256
    )

    # Export paths
    onnx_path = MODELS_DIR / 'model.onnx'
    quantized_path = MODELS_DIR / 'model_quantized.onnx'

    # Export model to ONNX
    print('Exporting ONNX model...')
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_path,
        opset_version=15,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'},
        },
        export_params=True,
        do_constant_folding=True,
    )
    print(f'ONNX model exported to {onnx_path}')

    # Quantize model
    print('Quantizing ONNX model...')
    quantize_dynamic(onnx_path, quantized_path, weight_type=QuantType.QUInt8)
    print(f'Quantized ONNX model created: {quantized_path}')
    model_reduction = 1 - quantized_path.stat().st_size / onnx_path.stat().st_size
    print(f'Model size reduced by {100 * model_reduction:.1f}%')

    # export tokenizer
    tokenizer.save_pretrained(MODELS_DIR)
    print(f'Tokenizer saved to {MODELS_DIR}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export trained model to ONNX format')
    parser.add_argument(
        '--model_path', type=str, required=True, help='Path to trained model directory'
    )
    args = parser.parse_args()

    export_onnx(model_path=args.model_path)
