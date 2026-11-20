# OnlineTrainingRL

OmniGen2 Flow-GRPO training with a frozen custom Qwen3-VL reward model, plus
GEdit inference for the original and post-training policies.

```text
OnlineTrainingRL/
  EditScore/       # Full upstream source snapshot, not a Git submodule
  running/         # Independent single-H200 training launcher/config
  inference/       # Independent base/checkpoint GEdit inference
```

EditScore source: https://github.com/VectorSpaceLab/EditScore, upstream commit
`4609c5d2ebb62fdebf665d3c924686d896ef1f74`. Its files and Apache-2.0 license
are preserved. The local EditScore checkout is also usable independently.

Training configuration: one H200 141GB shared by policy and reward, 12
rollouts/condition, 1 optimizer update/group, 200 updates, LoRA rank 32 / alpha
64, learning rate 5e-5, warmup 20. See [running/README.md](running/README.md).

The runners live outside EditScore. Set editscore_repo in each settings.json;
the default ../EditScore works when these directories are uploaded together.
Pretrained, reward checkpoint, source images and Conda environments remain
external and keep the configured HPC paths. No weights, dataset, logs or
generated images are included in this repository.

## Server location

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL
```

Configure running/settings.json and inference/settings.json before running.

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL/running
mkdir -p logs
sbatch run_train.slurm

cd ../inference
mkdir -p logs
sbatch run_omnigen2_base.slurm
sbatch run_omnigen2_checkpoint.slurm
```

Select the exact post-training checkpoint in inference/settings.json; it is
not selected automatically from the latest training run. See
[inference/README.md](inference/README.md) for outputs and resume behavior.

Use download_on_server.py to install/update source into the server directory
without requiring Git authentication. Existing settings.json files, outputs,
logs, caches and weights are preserved. The script only writes files present
in the source archive and does not recursively delete existing directories.
