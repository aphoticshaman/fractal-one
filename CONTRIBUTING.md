# Contributing to Fractal One

This project has been released to the community. There is no active maintainer. The codebase is entirely community-driven.

## How to Contribute

1. Fork the repository
2. Clone your fork
3. Build: `cargo build --features headless`
4. Test: `cargo test --features headless`
5. Make your changes
6. Submit a pull request

## Requirements

- Rust 1.75+
- Run `cargo fmt` and `cargo clippy` before submitting

## Code Style

- Follow `rustfmt` defaults
- Address clippy warnings
- Document public APIs
- No `.unwrap()` in non-test code

## Governance

Lazy consensus:
- PRs passing CI with no objections for 7 days may be merged
- Breaking changes require discussion in an issue first

## Communication

- GitHub Issues for bugs and features
- GitHub Discussions for questions
- GitHub Security Advisories for vulnerabilities

## License

Contributions are licensed under the same terms as the project.

---

This project belongs to the community. Make it yours.
