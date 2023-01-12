# Steps to set up the environment for testing

Set up conda environment:
```
conda create --name picai_baseline python=3.9
```

Activate environment:
```
conda activate picai_baseline
```

Install dependencies:
```
pip install -e .
pip install -r requirements_dev.txt
```

Perform tests:
```
pytest
mypy src
flake8 src
```

# Programming conventions
AutoPEP8 for formatting (this can be done automatically on save, see e.g. https://code.visualstudio.com/docs/python/editing)

# Push release to PyPI
1. Increase version in setup.py, and set below
2. Build: `python -m build`
3. Test package distribution: `python -m twine upload --repository testpypi dist/*0.8.0*`
4. Distribute package to PyPI: `python -m twine upload dist/*0.8.0*`
