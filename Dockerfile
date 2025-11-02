# Use official Python 3.12 slim image
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /code

# Copy pyproject.toml and uv.lock for uv dependencies
COPY pyproject.toml uv.lock /code/

# Install Python packages from uv.lock
RUN uv sync --frozen

# Install en_core_web_lg
RUN uv run python -m spacy download en_core_web_lg

# Copy application code
COPY ./helper_lib /code/helper_lib

# Clean up unnecessary files
RUN find . -name "*.pyc" -delete
RUN find . -name "__pycache__" -type d -exec rm -r {} +

# Command to run FastAPI with uv
CMD ["uv", "run", "fastapi", "run", "helper_lib/main.py", "--port", "80"]