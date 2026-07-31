.PHONY: build install run clean

build:
	cargo build --release

install: build
	mkdir -p ~/.local/bin
	install -m 755 ../target/release/cce-cloud ~/.local/bin/cce-cloud
	mkdir -p ~/.config/systemd/user
	install -m 644 cce-cloud.service ~/.config/systemd/user/cce-cloud.service

run:
	cargo run

clean:
	cargo clean
