from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
rknn.load_onnx(model='weights/best.onnx')
#IF NOT USING QUANTIZATION
rknn.build(do_quantization=False, dataset='dataset/dataset.txt')
rknn.export_rknn('weights/bestwithoutquantization.rknn')

#IF USING QUANTIZATION (MODEL MENJADI RUSAK)
#rknn.build(do_quantization=True, dataset='dataset/dataset.txt')
#rknn.export_rknn('weights/bestwithquantization.rknn')
rknn.release()
