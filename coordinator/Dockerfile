FROM rust:1.75 as builder

WORKDIR /app

# Copy proto files
COPY proto/ /app/proto/

# Copy coordinator source
COPY coordinator/Cargo.toml coordinator/Cargo.lock /app/coordinator/
COPY coordinator/build.rs /app/coordinator/
COPY coordinator/src/ /app/coordinator/src/

# Build
WORKDIR /app/coordinator
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/coordinator/target/release/coordinator /usr/local/bin/

EXPOSE 50051 9090

CMD ["coordinator"]
