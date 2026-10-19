# syntax=docker/dockerfile:1

# ---- builder: build the wheel from source -----------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps -w /dist .

# ---- runtime: install the wheel only, run as non-root ------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="react-planning-agent" \
      org.opencontainers.image.description="A bounded ReAct planning agent: observe, think, act, reflect - without infinite loops." \
      org.opencontainers.image.source="https://github.com/imranmali/react-planning-agent" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN groupadd --system --gid 10001 agent \
    && useradd --system --uid 10001 --gid agent --create-home --home-dir /home/agent \
       --shell /usr/sbin/nologin agent

COPY --from=builder /dist/ /tmp/dist/
RUN pip install /tmp/dist/*.whl && rm -rf /tmp/dist

USER 10001:10001
WORKDIR /home/agent

ENTRYPOINT ["react-agent"]
CMD ["--help"]
