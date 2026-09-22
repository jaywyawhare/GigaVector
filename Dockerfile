# syntax=docker/dockerfile:1
#
# Multi-stage build for the GigaVector HTTP server daemon (tools/gvserver.c).
# Built via CMake so the REST server (libmicrohttpd), TLS (OpenSSL) and webhook
# delivery (libcurl) are detected and linked. The GigaVector library is static,
# so the runtime image only needs the shared system deps.

FROM debian:bookworm-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        libmicrohttpd-dev libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
RUN cmake -S . -B build-docker -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build-docker --target gvserver -j"$(nproc)"

FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libmicrohttpd12 libcurl4 libssl3 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home-dir /data gigavector \
    && mkdir -p /data && chown gigavector:gigavector /data
COPY --from=build /src/build-docker/gvserver /usr/local/bin/gvserver
USER gigavector
ENV GV_DATA_DIR=/data \
    GV_PORT=8080 \
    GV_INDEX=hnsw \
    GV_DIMENSION=128
EXPOSE 8080
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${GV_PORT}/health" || exit 1
ENTRYPOINT ["gvserver"]
