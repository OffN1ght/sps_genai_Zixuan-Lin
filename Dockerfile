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

# # # Command to run FastAPI with uv
CMD ["uv", "run", "fastapi", "run", "helper_lib/main.py", "--port", "80"]

# # Use official Python 3.12 slim image
# FROM python:3.12-slim-bookworm

# # System deps for uv + general builds
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     curl ca-certificates \
#  && rm -rf /var/lib/apt/lists/*

# # Install uv (official way)
# ADD https://astral.sh/uv/install.sh /uv-installer.sh
# RUN sh /uv-installer.sh && rm /uv-installer.sh
# ENV PATH="/root/.local/bin/:$PATH"

# # uv docker best practice: avoid hardlink warnings + enable cache reuse
# ENV UV_LINK_MODE=copy

# WORKDIR /code

# # Copy only dependency files first, so this layer is cacheable
# COPY pyproject.toml uv.lock /code/

# # Install dependencies only (not the project) with cache mount
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked --no-install-project

# # ---- OPTIONAL: remove this if helper_lib doesn't need spaCy model ----
# # It makes builds MUCH slower and larger.
# # RUN --mount=type=cache,target=/root/.cache/uv \
# #     uv run python -m spacy download en_core_web_lg

# # Copy application code
# COPY ./helper_lib /code/helper_lib

# # If your project needs to be installed (usually not necessary for helper_lib),
# # run a final sync without touching deps:
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked

# # Clean pycache
# RUN find . -name "*.pyc" -delete \
#  && find . -name "__pycache__" -type d -exec rm -r {} +

# # Run FastAPI
# CMD ["uv", "run", "fastapi", "run", "helper_lib/main.py", "--host", "0.0.0.0", "--port", "80"]
