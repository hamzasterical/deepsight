import numpy as np
import pytest


class TestPredictorInit:
    def test_importable(self):
        from src.inference.predictor import Predictor
        assert Predictor is not None


class TestPredictorPredict:
    def test_predict_method_exists(self):
        from src.inference.predictor import Predictor
        assert hasattr(Predictor, "predict")

    def test_predict_from_bytes_method_exists(self):
        from src.inference.predictor import Predictor
        assert hasattr(Predictor, "predict_from_bytes")

    def test_predict_from_path_method_exists(self):
        from src.inference.predictor import Predictor
        assert hasattr(Predictor, "predict_from_path")




class TestExport:
    def test_export_functions_importable(self):
        from src.inference.export import export_and_quantize, export_to_onnx, quantize_onnx
        assert export_to_onnx is not None
        assert quantize_onnx is not None
        assert export_and_quantize is not None
