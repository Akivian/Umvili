# Contributing to Umvili

Thank you for your interest in contributing to Umvili! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Respect different viewpoints and experiences

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

- **Clear title and description**: What happened vs. what you expected
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Environment**: Python version, OS, and relevant package versions
- **Error messages**: Full error traceback if applicable
- **Screenshots**: If relevant to visualization issues

### Suggesting Features

Feature suggestions are welcome! Please:

- Check existing issues to avoid duplicates
- Provide a clear description of the feature
- Explain the use case and potential benefits
- Consider implementation complexity

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Follow the code style guidelines (see below)
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**:
   ```bash
   git commit -m "feat: add your feature description"
   ```
   Use conventional commit format (see below)
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**:
   - Provide a clear description
   - Reference related issues
   - Wait for review and address feedback

## Development Setup

See [Development Guide](docs/DEVELOPMENT.md) for detailed setup instructions.

Quick setup:
```bash
# Clone your fork
git clone https://github.com/your-username/Umvili.git
cd Umvili

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

## Code Style

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Maximum line length: 100 characters
- Use 4 spaces for indentation
- Use type hints for all function signatures
- Write docstrings for all public functions and classes

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Type Hints

Always use type hints:

```python
def create_agent(
    self,
    agent_type: str,
    x: int,
    y: int,
    agent_id: int,
    **kwargs: Any
) -> BaseAgent:
    """Create an agent instance."""
    pass
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_reward(
    self,
    agent: BaseAgent,
    action: int,
    next_state: np.ndarray
) -> float:
    """
    Calculate reward for an agent action.
    
    Args:
        agent: The agent performing the action
        action: Action taken by the agent
        next_state: Resulting state after action
        
    Returns:
        Reward value (float)
        
    Raises:
        ValueError: If agent or state is invalid
    """
    pass
```

## Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>: <short description>

<optional longer description>

<optional footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks (dependencies, config, etc.)
- `perf`: Performance improvements

### Examples

```
feat: add Q-value heatmap visualization

Add QValueHeatmapPanel component to visualize agent Q-values
on the environment map in real-time.

Closes #123
```

```
fix: correct boundary handling in RuleBasedAgent

Use observation.environment_size instead of global GRID_SIZE
constant for proper boundary checks.
```

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Add tests for bug fixes when applicable

## Documentation

- Update relevant documentation files
- Keep code examples up to date
- Document new features and APIs
- Use English for all code comments and documentation

## Project Structure

When adding new features:

- **New agent types**: Add to `src/core/agents.py` or `src/marl/`
- **New algorithms**: Add to `src/marl/`
- **Configuration**: Update `src/config/`
- **Visualization**: Update `src/utils/visualization.py`
- **Documentation**: Update relevant `.md` files in `docs/`

See [Architecture Documentation](docs/ARCHITECTURE.md) for details.

## Review Process

1. All PRs require at least one review
2. Address review comments promptly
3. Keep PRs focused and reasonably sized
4. Update your PR if requested

## Questions?

- Open an issue for questions
- Check existing documentation first
- Be patient - maintainers are volunteers

Thank you for contributing to Umvili! 🎉

