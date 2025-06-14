from rknn.api import RKNN

ONNX_MODEL = 'simplified_best.onnx'  # Update to simplified_best_v2.onnx if needed
DATASET_TXT = 'dataset.txt'
TARGET = 'rk3588'

def convert_for_target(target):
    print(f'--- Starting conversion for target: {target} ---')
    rknn = RKNN(verbose=True)  # Enable verbose logging

    rknn.config(
        target_platform=target,
        mean_values=[[0, 0, 0]],  # Replace with model-specific values if needed
        std_values=[[255, 255, 255]],  # Replace with model-specific values if needed
        quantized_dtype='asymmetric_quantized-8',
        output_optimize=True  # Changed to boolean
    )

    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print(f'[ERROR] Failed to load ONNX model for {target}')
        return

    ret = rknn.build(do_quantization=True, dataset=DATASET_TXT)
    if ret != 0:
        print(f'[ERROR] Failed to build RKNN model for {target}')
        return

    output_file = f'model_{target}.rknn'
    ret = rknn.export_rknn(output_file)
    if ret != 0:
        print(f'[ERROR] Failed to export RKNN model for {target}')
        return

    rknn.release()
    print(f'[SUCCESS] Model exported: {output_file}\n')

if __name__ == '__main__':
    convert_for_target(TARGET)