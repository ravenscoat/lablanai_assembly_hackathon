FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# App code
COPY configs/ configs/
COPY knowledge/ knowledge/
COPY scripts/ scripts/
COPY src/ src/

# Optional shared library (installed from parent context or pip), if you split
# common code into a package.
# RUN pip install /packages/shared

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8082

RUN chmod +x scripts/docker_entrypoint.sh

ENTRYPOINT ["scripts/docker_entrypoint.sh"]
CMD ["python", "src/main.py", "start"]
