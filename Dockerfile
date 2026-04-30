FROM tensorflow/tensorflow:2.14.0-gpu

WORKDIR /experiment

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/
ENV PATH="/experiment/.venv/bin:${PATH}"

COPY pyproject.toml README.md ./
COPY docker/entrypoint.sh /usr/local/bin/project-entrypoint

RUN chmod +x /usr/local/bin/project-entrypoint

ENTRYPOINT ["project-entrypoint"]
CMD ["python", "-m", "main"]
