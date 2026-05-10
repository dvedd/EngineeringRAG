FROM quay.io/docling-project/docling-serve-cu130:main

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng wget ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -q -O /usr/share/tesseract-ocr/5/tessdata/osd.traineddata \
    https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata/
ENV DOCLING_SERVE_ARTIFACTS_PATH=/models
ENV DOCLING_SERVE_ENABLE_UI=true
ENV DOCLING_SERVE_ENG_LOC_NUM_WORKERS=1
ENV DOCLING_SERVE_MAX_SYNC_WAIT=600
ENV UVICORN_WORKERS=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4

RUN mkdir -p /models && chown -R 1001:1001 /models
