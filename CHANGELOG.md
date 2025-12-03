# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Enhanced environment simulation (multi-resource system, obstacles, dynamic events)
- Deep learning state monitoring (Q-value heatmaps, attention visualization)
- Interactive experiment configuration system
- Data export and analysis tools
- Advanced visualization enhancements (interactive charts, 3D visualization)

See [Project Roadmap](docs/zh/PROJECT_ROADMAP.md) for details.

## [1.0.0] - 2024-12

### Added
- **Multi-line Chart Component**: Advanced `MultiLineChart` component supporting multiple data series, dynamic line management, and performance optimizations
- **Training Metrics Visualization**:
  - Training Loss curves (per algorithm type)
  - Q-Value Trend charts
  - TD Error visualization
  - Exploration Rate curves
- **Behavior Analysis Visualization**:
  - Action Distribution panel with semantic action descriptions
  - Reward Trend charts
  - Policy Entropy curves
- **View System**: Multi-view interface with Overview, Training, Behavior, and Debug views
- **Agent Distribution Panel**: Enhanced visualization showing agent types, counts, proportions, and average sugar levels
- **Unified Reward Calculator**: Centralized reward calculation system (`RewardCalculator`) for consistent reward computation
- **Configuration Management**: Comprehensive configuration system supporting JSON/YAML files with validation
- **Performance Monitoring**: Real-time FPS and performance metrics tracking

### Changed
- Refactored visualization system with modular component architecture
- Improved layout management with grid-based chart positioning
- Enhanced error handling across all agent classes
- Optimized state processing and local view calculations
- Updated documentation structure and organization

### Fixed
- Fixed `BaseAgent.reset()` configuration reference error
- Fixed `RuleBasedAgent` boundary handling for different environment sizes
- Fixed IQL and QMIX agent state update timing issues
- Improved reward calculation accuracy

### Performance
- Optimized visualization rendering with update frequency control
- Added sliding window data management (200 data points)
- Improved memory usage with efficient data structures

## [0.9.0] - 2024-11

### Added
- Initial MARL algorithm implementations (IQL, QMIX)
- Basic visualization system
- Rule-based agent implementations
- Sugar environment simulation

### Changed
- Initial project structure

---

## Version History

- **1.0.0**: First stable release with comprehensive visualization and monitoring
- **0.9.0**: Initial development version

## Notes

- Dates are approximate
- See commit history for detailed changes
- Breaking changes will be clearly marked

