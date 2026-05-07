# Contributing to Evo-AMR

We welcome contributions to the Evo-AMR project! This document provides guidelines for contributing to ensure a smooth collaboration process.

## 🤝 Ways to Contribute

- **Bug Reports**: Report issues or unexpected behavior
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit bug fixes, new features, or optimizations
- **Documentation**: Improve documentation, tutorials, or examples
- **Testing**: Add or improve test coverage

## 🚀 Getting Started

### Development Setup

1. **Fork the Repository**
   ```bash
   git clone https://github.com/haleyyy2001/Evo-Amr.git
   cd Evo-Amr
   ```

2. **Create Development Environment**
   ```bash
   conda env create -f environment.yml
   conda activate evo-amr
   pip install -e .
   ```

3. **Install Development Dependencies**
   ```bash
   pip install black isort flake8 pytest pytest-cov
   ```

### Code Style

We follow PEP 8 with some modifications:

- **Line Length**: 88 characters (Black default)
- **Import Sorting**: Use isort for import organization
- **Formatting**: Use Black for consistent code formatting
- **Linting**: Use flake8 for code quality checks

```bash
# Format code
black src tests utils
isort src tests utils

# Check code quality
flake8 src tests utils
```

### Testing

All contributions should include appropriate tests:

```bash
# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

## 📝 Contribution Process

### 1. Create an Issue

For bug reports or feature requests, please create an issue first to discuss:

- **Bug Reports**: Include steps to reproduce, expected behavior, and environment details
- **Feature Requests**: Describe the feature, use case, and potential implementation

### 2. Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-number
   ```

2. **Make Changes**
   - Follow code style guidelines
   - Add/update tests as needed
   - Update documentation if necessary

3. **Test Your Changes**
   ```bash
   # Run tests
   python -m pytest tests/

   # Test specific functionality
   python -m pytest tests/test_your_feature.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new embedding pooling method"
   # or
   git commit -m "fix: resolve memory leak in diagnostic pipeline"
   ```

   Use conventional commit messages:
   - `feat:` New features
   - `fix:` Bug fixes
   - `docs:` Documentation changes
   - `test:` Adding or modifying tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### 3. Pull Request Guidelines

When creating a pull request:

- **Title**: Clear and descriptive
- **Description**: Explain what changes were made and why
- **Testing**: Describe how the changes were tested
- **Documentation**: Update relevant documentation
- **Breaking Changes**: Highlight any breaking changes

#### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (describe):

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── test_config_manager.py    # Configuration tests
├── test_project_metadata.py  # Repository health checks
└── conftest.py               # Shared fixtures, when needed
```

### Writing Tests

1. **Unit Tests**: Test individual functions/methods
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows

Example test:
```python
import pytest
from config.config_manager import ConfigManager

def test_config_manager_resolves_model_path():
    config = ConfigManager()

    assert config.get_model_path("evo_1_131k_base")
```

## 📚 Documentation

### Documentation Types

1. **Code Documentation**: Docstrings for all public functions/classes
2. **User Documentation**: README, tutorials, examples
3. **API Documentation**: Automatic generation from docstrings
4. **Developer Documentation**: Contributing guidelines, architecture

### Docstring Format

Use Google-style docstrings:

```python
def extract_embeddings(self, sequences: List[str], sequence_ids: List[str]) -> Tuple[torch.Tensor, List[int]]:
    """Extract embeddings from a batch of sequences.

    Args:
        sequences: List of DNA sequences to process
        sequence_ids: Corresponding sequence identifiers

    Returns:
        Tuple containing:
            - embeddings: Extracted embeddings tensor [batch_size, seq_len, hidden_dim]
            - valid_lengths: List of valid sequence lengths

    Raises:
        ValueError: If sequences and sequence_ids have different lengths

    Example:
        >>> extractor = EmbeddingExtractor(config, device)
        >>> sequences = ["ATCGATCG", "GCTAGCTA"]
        >>> ids = ["seq1", "seq2"]
        >>> embeddings, lengths = extractor.extract_embeddings(sequences, ids)
    """
```

## 🐛 Bug Reports

When reporting bugs, please include:

### Environment Information
- Python version
- PyTorch version
- CUDA version (if using GPU)
- Operating system
- Hardware specifications

### Reproduction Steps
1. Clear steps to reproduce the issue
2. Expected behavior
3. Actual behavior
4. Error messages or logs

### Example Bug Report
```markdown
**Environment:**
- Python 3.11.0
- PyTorch 2.0.1
- CUDA 11.8
- Ubuntu 22.04
- RTX 4090

**Bug Description:**
Memory leak in embedding generation when processing large batches

**Reproduction Steps:**
1. Run embedding_generator.py with batch_size=32
2. Monitor GPU memory usage
3. Memory usage increases with each batch

**Expected Behavior:**
Memory usage should remain stable

**Actual Behavior:**
GPU memory increases until OOM error

**Error Message:**
RuntimeError: CUDA out of memory...
```

## 🚀 Feature Requests

When requesting features:

1. **Use Case**: Describe the problem or need
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other ways to solve the problem
4. **Implementation**: Any thoughts on implementation
5. **Breaking Changes**: Would this require breaking changes?

## 📋 Code Review Process

All contributions go through code review:

1. **Automated Checks**: CI/CD runs tests and style checks
2. **Manual Review**: Maintainers review code quality and design
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approval required for merge

### Review Criteria
- **Functionality**: Does it work as intended?
- **Code Quality**: Is it well-written and maintainable?
- **Performance**: Does it impact performance?
- **Documentation**: Is it properly documented?
- **Tests**: Are there adequate tests?

## 🎯 Priority Areas

We especially welcome contributions in:

- **Performance Optimization**: GPU acceleration, memory efficiency
- **Model Support**: New model architectures or variants
- **Visualization**: Better plots and analysis tools
- **Documentation**: Tutorials, examples, API docs
- **Testing**: Increase test coverage
- **Configuration**: Improve config system flexibility

## 📞 Getting Help

If you need help with contributing:

- **GitHub Discussions**: For questions and discussions
- **Issues**: For bug reports and feature requests
- **Email**: ht2666@columbia.edu for private inquiries

## 🏆 Recognition

Contributors will be recognized in:

- **Contributors List**: README.md
- **Release Notes**: Major contributions highlighted
- **Documentation**: Author attribution where appropriate

Thank you for contributing to Evo-AMR! 🙏
