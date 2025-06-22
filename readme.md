# Rust-crypto
Rust cryptographic functions that can be imported in python.

# Run test
Compile and run all test
> ./run_tests.sh

Run a specific test
> RUST_BACKTRACE=1 ./run_tests.sh test_expression_bin -- --nocapture

# Compile and install
Will compile and install the python package
> ./compile

# Flamegraph
Enabling perf for use by unprivileged users
> echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid

Generate the flamegraph based on the test
> RUSTFLAGS="-C target-cpu=native" flamegraph  -o flamegraph.svg -- cargo test test_aes_5_rounds_flame --release
