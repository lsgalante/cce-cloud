.PHONY: build install run clean

build:
	cargo build --release

install: build
	mkdir -p ~/.local/bin
	install -m 755 ../target/release/cce-cloud ~/.local/bin/cce-cloud

run:
	cargo run

clean:
	cargo clean
