//! Thin control surface binary (`marceline`) for the daemon.

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let verbose = args.iter().any(|a| a == "--verbose");

    if args.get(1).map(String::as_str) == Some("--version") {
        println!("marceline {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    marceline_core::logging::init(verbose);
    tracing::info!(version = env!("CARGO_PKG_VERSION"), "marceline starting");
    tracing::debug!(verbose, "verbose logging enabled by --verbose flag");

    println!("marceline {}", env!("CARGO_PKG_VERSION"));
}
