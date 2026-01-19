# Contributing to Customer Churn Prediction System

Thank you for considering contributing to this project! 🎉

## 🚀 Quick Start for Contributors

1. **Fork the repository**
2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/churn_e2e_ml.git
   cd churn_e2e_ml
   ```
3. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```
4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
5. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📋 Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Keep functions small and focused

### Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR

```bash
pytest tests/ -v
```

### Commit Messages

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code formatting
- `refactor:` Code restructuring
- `test:` Adding tests

Example: `feat: add email notification for high-risk customers`

## 🔧 Areas for Contribution

### High Priority

- [ ] Model monitoring with Evidently AI
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Kubernetes deployment manifests
- [ ] Feature store integration (Feast)

### Medium Priority

- [ ] Additional ML models (LightGBM, CatBoost)
- [ ] A/B testing framework
- [ ] Real-time streaming with Kafka
- [ ] Email alerts for high-risk customers

### Documentation

- [ ] API usage examples
- [ ] Video tutorials
- [ ] Architecture diagrams

## 📝 Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt if you add new dependencies
3. Ensure all tests pass
4. Request review from maintainers

## 💬 Questions?

Feel free to open an issue for any questions or suggestions!

---

Thank you for contributing! 🙏
