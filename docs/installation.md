# Installing RL_WEF_MS

[Back to project](../README.md)

## 1. Download the project

Install Python **3.12** and Git, then clone:

```bash
git clone https://github.com/Claudio-zb/RL_WEF_MS.git
cd RL_WEF_MS
```

Alternatively, use GitHub **Code → Download ZIP**, extract the archive and open a terminal in the extracted folder (usually `RL_WEF_MS-main`). Keep the included data and checkpoint folders.

## 2. Create an isolated Python environment

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, invoke `.\.venv\Scripts\python.exe` directly in place of `python` in the following commands. No change to system execution policy is needed.

## 3. Install the dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The versioned [requirements.txt](../requirements.txt) includes the numerical libraries, PyTorch, Stable-Baselines3, Gymnasium, CasADi, plotting libraries and the FFmpeg package for MP4 export. Downloads can be large because PyTorch is included. The supported commands default to CPU; a GPU and external LaTeX installation are not required for the quick start.

To import the project from another directory, optionally install the checkout in editable mode after installing requirements:

```bash
python -m pip install --no-deps -e .
```

The distribution name is `rl-wef-ms`. The Python namespace remains `rl_ems` to preserve existing scripts and commands. Keep the repository and its data in place; the project is designed to run from this checkout.

## 4. Verify your installation

From the project root:

```bash
python -m unittest discover -s tests -v
python -m rl_ems.forecasting.predict --day 64 --output outputs/first-forecast.csv
python -m rl_ems.simulation.run --days 2 --output outputs/first-simulation
```

The first command runs the regression checks. The others create a forecast CSV and a small coupled simulation. To repeat a command, choose a new output path; existing results are preserved.

## Common setup issues

- **`No module named ...`**: activate the virtual environment and run `python -m pip install -r requirements.txt` using that same interpreter.
- **No matching package version**: check `python --version`; these dependency versions were validated with Python 3.12. Use the commands above to create the environment with that version.
- **Data/checkpoint not found**: run from the extracted/cloned project root and check that `environments/Data`, `predictive_models` and `logs` are present.
- **CUDA or LaTeX errors in an experiment**: original files under `experiments/` retain their historical assumptions. Start with the documented `rl_ems` commands; see the [research notes](reproducibility.md) before running those original scripts.
