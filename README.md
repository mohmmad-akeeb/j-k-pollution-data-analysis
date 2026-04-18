# Polution Project

## Setup Python Environment

1. Open a terminal in the project root:
   ```bash
   cd <working_dir>
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   - Linux/macOS:
     ```bash
     source venv/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```

4. Install the project dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running Jupyter Notebook

1. With the virtual environment active, start Jupyter:
   ```bash
   jupyter notebook
   ```

2. In your browser, open the notebook file you want to edit.

3. When finished, stop the server by pressing `Ctrl+C` in the terminal.

## Notes

- The project already ignores `venv/`, `__pycache__/`, and `.ipynb_checkpoints/` in `.gitignore`.
- If you need to install any additional packages, add them to `requirements.txt` and run:
  ```bash
  pip install <package>
  pip freeze > requirements.txt
  ```
