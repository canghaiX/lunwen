FROM inferkernellab:cuda

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.cache \
    TRANSFORMERS_OFFLINE=1 \
    REVIMPACT_MODEL_PATH=/models/Qwen/Qwen3-0.6B

WORKDIR /workspace/lunwen
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY scripts ./scripts
RUN pip install --no-cache-dir --no-index -e '.[ml,dev]' || \
    pip install --no-cache-dir -e '.[ml,dev]'
RUN mkdir -p data outputs
CMD ["bash"]
