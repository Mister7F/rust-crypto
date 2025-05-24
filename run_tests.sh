rustfmt ./src/*.rs && RUSTFLAGS="-C target-cpu=native" cargo test --release "$@"
