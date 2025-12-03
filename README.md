# Umvili

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-PEP%208-brightgreen.svg)](https://www.python.org/dev/peps/pep-0008/)

A sandbox-style multi-agent reinforcement learning (MARL) algorithm comparison platform with real-time visualization.

## Overview

Umvili is a visualization platform for researching and comparing different multi-agent reinforcement learning algorithms. It provides an interactive environment where you can observe, compare, and analyze the behavior of various MARL algorithms in real-time.

### Key Features

- 🎯 **Multiple Algorithms**: Support for IQL, QMIX, and rule-based agents (conservative, exploratory, adaptive)
- 📊 **Real-time Visualization**: Dynamic display of agent behavior, sugar distribution, and learning curves
- ⚙️ **Flexible Configuration**: Support for JSON/YAML configuration files and command-line arguments
- 🔄 **Algorithm Comparison**: Run multiple algorithms simultaneously for performance comparison
- 📈 **Performance Monitoring**: Real-time monitoring of FPS, learning progress, and other metrics

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

For detailed project structure, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Documentation

- [Architecture Documentation](docs/ARCHITECTURE.md) - Project structure and design
- [Configuration Guide](docs/CONFIGURATION.md) - Configuration management
- [Development Guide](docs/DEVELOPMENT.md) - Development setup and guidelines
- [中文文档](docs/zh/README.md) - Chinese documentation

## Configuration

The project supports flexible configuration through:

- **Configuration files**: JSON/YAML format (see `config/default.json`)
- **Command-line arguments**: Override specific settings
- **Code configuration**: Programmatic configuration

For detailed configuration options, see [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Development

### Code Style

- Follow PEP 8 code style
- Use type hints
- Complete docstrings for all public APIs

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

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
