.PHONY: build install run clean

build:
	cargo build --release

install: build
	mkdir -p ~/.local/bin
	install -m 755 ../target/release/clear-cloud ~/.local/bin/clear-cloud

run:
	cargo run

clean:
	cargo clean
