FROM ubuntu:22.04 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    autoconf \
    libtool \
    libssl-dev \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install gRPC and protobuf
RUN git clone --recurse-submodules -b v1.58.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc /var/local/git/grpc && \
    cd /var/local/git/grpc && \
    mkdir -p cmake/build && \
    cd cmake/build && \
    cmake -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ../.. && \
    make -j$(nproc) && \
    make install

WORKDIR /app

# Copy source
COPY proto/ /app/proto/
COPY aggregator/ /app/aggregator/

# Build
WORKDIR /app/aggregator
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Runtime image
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /app/aggregator/build/aggregator_server /usr/local/bin/

RUN ldconfig

EXPOSE 50052

CMD ["aggregator_server"]
