# Run Instructions
## Benchmark/Training Example
Use `export PYTHONPATH=".:$PYTHONPATH"` to add all the modules to the path.

## Adding Jupyter Kernels
To add Jupyter kernels from your project, simply do:
```
# 1. Activate your virtual environment
source /path/to/venv/bin/activate      

# 2. Install the Jupyter kernel package inside that venv
pip install ipykernel

# 3. Add it as a new Jupyter kernel
python -m ipykernel install --user --name=venv_name --display-name "Python (venv_name)"
```
And you can verify this through:
```
jupyter kernelspec list
```