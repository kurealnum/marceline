//! Thin control surface binary (`marceline`) for the daemon.

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.get(1).map(String::as_str) == Some("--version") {
        println!("marceline {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    println!("marceline {}", env!("CARGO_PKG_VERSION"));
}
