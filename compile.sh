# This rust flag is super important for performance
# >>> RUSTFLAGS="-C target-cpu=native"
# equivalent to "-march=native" in C++
rustfmt ./src/*.rs && \
    RUSTFLAGS="-C target-cpu=native" maturin build --release && \
    python3 -m pip install --force-reinstall ./target/wheels/rust_crypto*.whl --break-system-packages
