#!/bin/bash
# Archive one finished Slurm job's raw data, read-only, with provenance.
#   bench/archive_job.sh <jobid> <name> <file-or-dir>...
# Creates data/<jobid>_<name>/ containing copies of the given outputs plus
#   job_script.sh  exact batch script Slurm ran (sacct -B)
#   sacct.txt      node list, start/end, elapsed, state
#   provenance.txt sha256 of source, binary and bench scripts at archive time
# and appends every archived file to data/MANIFEST.sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
jobid=$1; name=$2; shift 2
dest=data/${jobid}_${name}
if [ -e "$dest" ]; then
    echo "refusing to overwrite existing $dest" >&2
    exit 1
fi
state=$(sacct -nX -j "$jobid" -o State | tr -d ' ')
case "$state" in
    COMPLETED|FAILED|TIMEOUT|CANCELLED*) ;;
    *) echo "job $jobid is $state, not finished" >&2; exit 1 ;;
esac
mkdir -p "$dest"
for f in "$@"; do cp -a "$f" "$dest"/; done
sacct -B -j "$jobid" | sed '1,2d' > "$dest/job_script.sh"
sacct -j "$jobid" -o JobID,JobName,Partition,NodeList,NCPUS,Start,End,Elapsed,State,ExitCode -P \
    > "$dest/sacct.txt"
{
    echo "archived: $(date -Is) by $USER on $(hostname)"
    echo "note: hashes are of the files at archive time; the binary was built"
    echo "      before the job if its mtime precedes the job start in sacct.txt"
    for f in src/*.c src/*.h poisson tests/test_coarse tests/*.c tests/*.py Makefile env.sh bench/*.sh tests/*.sh; do
        [ -f "$f" ] && echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c %y "$f")  $f"
    done
} > "$dest/provenance.txt"
find "$dest" -type f | sort | xargs sha256sum >> data/MANIFEST.sha256
chmod -R a-w "$dest"
echo "archived $jobid -> $dest ($(find "$dest" -type f | wc -l) files)"
