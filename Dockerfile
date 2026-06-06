# Physics MCP Server Dockerfile
# ===============================
# Multi-stage build for optimal image size
# Based on chuk-mcp-server patterns

# Build stage
FROM python:3.14-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project configuration
COPY pyproject.toml README.md ./
COPY src ./src

# Install the package with all dependencies
# Use --no-cache to reduce layer size
RUN uv pip install --system --no-cache -e .

# Runtime stage
FROM python:3.14-slim

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app/src ./src
COPY --from=builder /app/README.md ./
COPY --from=builder /app/pyproject.toml ./

# Create non-root user for security
RUN useradd -m -u 1000 mcpuser && \
    chown -R mcpuser:mcpuser /app

# Switch to non-root user
USER mcpuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); import chuk_mcp_physics; print('OK')" || exit 1

# Default command - run MCP server in HTTP mode for Docker
CMD ["python", "-m", "chuk_mcp_physics.server", "http"]

# Expose port for HTTP mode (if using HTTP transport)
EXPOSE 8000

# Labels for metadata
LABEL maintainer="chris@example.com" \
      description="Physics MCP Server - Rapier-based physics simulations and calculations" \
      version="0.1.0" \
      org.opencontainers.image.source="https://github.com/yourusername/chuk-mcp-physics" \
      org.opencontainers.image.title="Physics MCP Server" \
      org.opencontainers.image.description="MCP server for physics simulations using Rapier" \
      org.opencontainers.image.authors="Chris Hay <chris@example.com>"
