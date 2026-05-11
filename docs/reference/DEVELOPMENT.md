# Development Guide

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Setup Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/Akivian/Umvili.git
   cd Umvili
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python main.py --help
   ```

## Code Style

### PEP 8 Compliance

Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines:

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable and function names
- Follow naming conventions:
  - Functions: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`

### Type Hints

Use type hints for all function signatures:

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

## Project Structure

### Adding New Features

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement feature**
   - Follow existing code patterns
   - Add type hints and docstrings
   - Write tests if applicable

3. **Test your changes**
   ```bash
   python main.py --config config/default.json
   ```

4. **Commit changes**
   ```bash
   git add .
   git commit -m "Add: description of your feature"
   ```

5. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Adding New Agent Types

1. **Create agent class** in `src/core/agents.py`:
   ```python
   class YourAgent(BaseAgent):
       """Your agent description."""
       
       def __init__(self, x: int, y: int, agent_id: int, **kwargs):
           super().__init__(x, y, agent_id)
           # Your initialization
       
       def act(self, observation: np.ndarray) -> int:
           # Your action logic
           pass
   ```

2. **Register in factory** (`src/core/agent_factory.py`):
   ```python
   factory.register_agent_type("your_agent", YourAgent)
   ```

3. **Add configuration** (`src/config/agent_config.py`):
   ```python
   @dataclass
   class YourAgentConfig(BaseAgentConfig):
       your_param: float = 1.0
   ```

4. **Update defaults** (`src/config/defaults.py`):
   ```python
   DEFAULT_AGENT_CONFIGS["your_agent"] = {
       "your_param": 1.0
   }
   ```

### Adding New MARL Algorithms

1. **Create algorithm module** in `src/marl/`:
   ```python
   class YourMARLAgent(LearningAgent):
       """Your MARL algorithm."""
       pass
   ```

2. **Implement required methods**:
   - `act()`: Action selection
   - `learn()`: Learning logic
   - `update()`: Parameter updates

3. **Register in factory** and add configuration (same as above)

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_agents.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

Create test files in `tests/` directory:

```python
import pytest
from src.core.agents import RuleBasedAgent

def test_agent_creation():
    """Test agent creation."""
    agent = RuleBasedAgent(x=10, y=10, agent_id=1)
    assert agent.x == 10
    assert agent.y == 10
```

## Debugging

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Debugging Techniques

1. **Print statements**: Use `logger.debug()` instead of `print()`
2. **Breakpoints**: Use IDE debugger or `pdb`
3. **Configuration**: Test with minimal configuration first

## Documentation

### Updating Documentation

- Update relevant `.md` files in `docs/`
- Keep code examples up to date
- Document new features and APIs

### Code Comments

- Use English for all code comments
- Explain "why" not "what"
- Keep comments concise and relevant

## Git Workflow

### Commit Messages

Follow conventional commit format:

```
type: short description

Longer description if needed

- Bullet point for changes
- Another bullet point
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance tasks

### Branch Naming

- `feature/`: New features
- `fix/`: Bug fixes
- `docs/`: Documentation updates
- `refactor/`: Code refactoring

## Performance Optimization

### Profiling

```python
import cProfile
cProfile.run('your_function()')
```

### Best Practices

1. Avoid premature optimization
2. Profile before optimizing
3. Use NumPy for numerical operations
4. Cache expensive computations

## Contributing

### Pull Request Process

1. Fork the repository
2. Create your feature branch
3. Make your changes
4. Add/update tests
5. Update documentation
6. Submit pull request with clear description

### Code Review

- All PRs require review
- Address review comments promptly
- Keep PRs focused and small when possible

## Resources

- [Python Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Type Hints](https://docs.python.org/3/library/typing.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

For questions or issues, please open an issue on GitHub.

