### Installation

From the project's root directory,

```
git clone https://github.com/espnet/espnet

cd espnet/tools
./setup_anaconda.sh miniconda espnet 3.11

make TH_VERSION=2.6.0 CUDA_VERSION=12.6
. ./activate_python.sh; python3 check_install.py

pip install --user loralib "datasets<4.0.0"
```

### Usage

To train an adapter for a single language,

```
sbatch -p GPU-shared --gres=gpu:v100:1 -t 4:00:00 scripts/adapters.sh {FLEURS language code} {OWSM language code}
```

To evaluate the baseline model for a language,

```
sbatch -p GPU-shared --gres=gpu:v100:1 -t 1:00:00 scripts/test_base.sh {FLEURS language code} {OWSM language code}  {Optional: num_samples}
```
