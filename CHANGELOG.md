# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Airfoil` class that stores validated surface coordinates and computes geometric properties.
- `Scalar` type alias for NumPy floating-point scalars.
- Generic `Vector[T]`, `Matrix[T]`, and `Tensor[T]` type aliases for NumPy arrays with floating-point elements.

- CI workflow that builds the package once, uploads it as an artifact, and runs the test suite against the installed wheel on Ubuntu, Windows, and macOS with Python 3.13, 3.14, and 3.15.
- Release workflow that runs CI, checks that the tag matches the project version, and creates a GitHub release for `vX.Y.Z` tags on `master`.
- Test coverage reporting with a 90% minimum threshold, plus JUnit XML and HTML test reports.

### Changed

### Deprecated

### Removed

### Fixed

### Security

[Unreleased]: https://github.com/CFBaptista/WingSection/commits/master
