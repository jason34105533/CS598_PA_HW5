# Archived experiment data

Read-only raw data, one directory per Slurm job: `data/<jobid>_<name>/`,
created by `bench/archive_job.sh`; never edit these files. Each directory
holds the raw job output, `job_script.sh` (the exact batch script Slurm ran,
from `sacct -B`), `sacct.txt` (nodes, start/end, state) and
`provenance.txt` (sha256 of sources, binaries and scripts at archive time).
`MANIFEST.sha256` lists every archived file:

    sha256sum -c data/MANIFEST.sha256

All jobs: `eng-instruction` partition, exclusive AE7713 nodes
(2x AMD EPYC 7713, 128 cores), gcc 13.3.0 + Open MPI 5.0.1, `-O3 -march=znver3`.

| Job | Name | Nodes | What | Outcome |
|---|---|---|---|---|
