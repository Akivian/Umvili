# Umvili

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)

A sandbox-style multi-agent reinforcement learning (MARL) algorithm comparison platform with real-time visualization.

## Overview

Umvili is a visualization platform for researching and comparing different multi-agent reinforcement learning algorithms. It provides an interactive environment where you can observe, compare, and analyze the behavior of various MARL algorithms in real-time.

The platform features a sophisticated multi-resource environment with dynamic resource generation and hazard zones, enabling complex multi-agent scenarios where agents must balance resource collection, competition, and risk avoidance.

### Key Features

- 🎯 **Multiple Algorithms**: Support for IQL, QMIX, and rule-based agents (conservative, exploratory, adaptive)
- 🌍 **Complex Multi-Resource Environment**:
  - **Sugar**: Basic resource with continuous regeneration across the map
  - **Spice**: High-value rare resource that spawns in 1–2 small concentrated areas, depletes, and respawns after a delay in new safe regions
  - **Hazard**: Dynamic danger zones evolving from a single random core point. The core region grows gradually and forms a large connected danger area that stabilizes around ~1/3 of the map with a “breathing” boundary, permanently clearing resources and dealing configurable damage per step to agents.
- 📊 **Advanced Visualization**: 
  - Real-time training metrics (Loss, Q-value, TD Error, Exploration Rate)
  - Behavior analysis (Action Distribution, Reward Trends, Policy Entropy)
  - Q-value heatmaps overlaid on the environment map
  - Network internal state visualization (policy distribution, entropy, gradient norms)
  - Multi-view interface (Overview, Training, Behavior, Debug, Network)
  - Interactive charts with multi-line support
- ⚙️ **Flexible Configuration**: Support for JSON/YAML configuration files and command-line arguments
- 🔄 **Algorithm Comparison**: Run multiple algorithms simultaneously for performance comparison
- 📈 **Performance Monitoring**: Real-time monitoring of FPS, learning progress, and other metrics
- 🧠 **Deep Learning Integration**: PyTorch-based neural networks with experience replay

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9.0+
- Pygame 2.1.0+

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akivian/Umvili.git
   cd Umvili
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

## Usage

### Basic Usage

```bash
# Run with default configuration
python main.py

# Run with custom configuration file
python main.py --config config/default.json

# Run with command-line arguments
python main.py --simulation-type comparative --grid-size 100 --agents 200
```

### Supported Agent Types

- `rule_based`: Rule-based agent
- `conservative`: Conservative agent
- `exploratory`: Exploratory agent
- `adaptive`: Adaptive agent
- `iql`: Independent Q-Learning algorithm
- `qmix`: QMIX algorithm

## Project Structure

```
Umvili/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── config/                 # Configuration files
│   └── default.json
└── src/                    # Source code
    ├── config/             # Configuration management
    ├── core/               # Core modules (agents, environment, simulation)
    ├── marl/               # MARL algorithm implementations
    └── utils/             # Utility modules
```

For detailed project structure, see [docs/reference/ARCHITECTURE.md](docs/reference/ARCHITECTURE.md).

## Documentation

- [Architecture Documentation](docs/reference/ARCHITECTURE.md) - Project structure and design
- [Configuration Guide](docs/reference/CONFIGURATION.md) - Configuration management
- [Development Guide](docs/reference/DEVELOPMENT.md) - Development setup and guidelines
- [Version Control Guide](docs/reference/VERSION_CONTROL_GUIDE.md) - Git workflow and best practices
- [中文文档](docs/zh/README.md) - Chinese documentation
- [项目路线图](docs/zh/PROJECT_ROADMAP.md) - Chinese project roadmap

## Configuration

The project supports flexible configuration through:

- **Configuration files**: JSON/YAML format (see `config/default.json`)
- **Command-line arguments**: Override specific settings
- **Code configuration**: Programmatic configuration

For detailed configuration options, see [docs/reference/CONFIGURATION.md](docs/reference/CONFIGURATION.md).

## Development

### Code Style

- Follow PEP 8 code style
- Use type hints
- Complete docstrings for all public APIs

### Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

For quick reference:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Akivian**

- GitHub: [@Akivian](https://github.com/Akivian)

## Acknowledgments

- Built with [PyTorch](https://pytorch.org/) for deep learning
- Visualization powered by [Pygame](https://www.pygame.org/)

## Citation

If you use Umvili in your research, please cite:

```bibtex
@software{umvili2025,
  title={Umvili: A MARL Algorithm Comparison Platform},
  author={Akivian},
  year={2025},
  url={https://github.com/Akivian/Umvili}
}
```

---

**Note**: This project is under active development. Features and APIs may change.
