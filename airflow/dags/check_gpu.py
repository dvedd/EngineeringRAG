from airflow.sdk import dag, task
from pendulum import datetime


@dag(
    dag_id="check_gpu_environment",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["debug"],
)
def check_gpu():

    @task
    def check_cuda_torch():
        import torch

        result = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "N/A",
            "cuda_version": torch.version.cuda,
        }
        import logging

        logging.info(f">>> Torch CUDA: {result}")
        return result

    @task
    def check_onnxruntime():
        import onnxruntime as ort

        providers = ort.get_available_providers()
        import logging

        logging.info(f">>> ORT providers: {providers}")
        assert "CUDAExecutionProvider" in providers, (
            f"CUDA недоступен! Доступны только: {providers}"
        )
        return providers

    @task
    def check_fastembed_gpu(providers: list):
        import logging

        from fastembed import TextEmbedding

        if isinstance(providers, str):
            import json

            providers = json.loads(providers)

        model = TextEmbedding(
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            providers=providers,
        )
        vec = list(model.embed(["passage: тест GPU"]))
        assert len(vec) == 1
        assert len(vec[0]) == 768
        logging.info(f">>> fastembed GPU OK, vec size={len(vec[0])}")
        return "OK"

    @task
    def check_colbert_gpu(providers: list):
        import logging

        from fastembed import LateInteractionTextEmbedding

        model = LateInteractionTextEmbedding(
            "colbert-ir/colbertv2.0",
            providers=providers,
        )
        vec = list(model.embed(["passage: тест ColBERT"]))
        assert len(vec) == 1
        logging.info(f">>> ColBERT GPU OK, tokens={len(vec[0])}, dim={len(vec[0][0])}")
        return "OK"

    # граф
    torch_info = check_cuda_torch()
    ort_providers = check_onnxruntime()
    dense_ok = check_fastembed_gpu(ort_providers)
    colbert_ok = check_colbert_gpu(ort_providers)
    torch_info
    ort_providers >> [dense_ok, colbert_ok]


check_gpu()
