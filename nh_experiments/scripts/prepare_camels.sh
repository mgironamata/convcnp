#!/bin/bash
# Verify, unpack and arrange CAMELS-US for neuralhydrology.
#
#   bash nh_experiments/scripts/prepare_camels.sh [/scratch2/mg963/data/camels_us]
#
# Expects in <root>/raw/ the Zenodo files listed in <root>/raw/files.tsv (see raw/download.sh):
#   basin_timeseries_v1p2_metForcing_obsFlow.zip, camels_*.txt (attributes v2.0),
#   optionally basin_timeseries_v1p2_modelOutput_daymet.zip (SAC-SMA benchmark).
# Produces <root>/CAMELS_US/ with the layout neuralhydrology's `camels_us` dataset expects:
#   basin_mean_forcing/daymet/<huc>/<id>_lump_cida_forcing_leap.txt
#   usgs_streamflow/<huc>/<id>_streamflow_qc.txt
#   camels_attributes_v2.0/camels_{clim,geol,hydro,name,soil,topo,vege}.txt
set -euo pipefail
ROOT=${1:-/scratch2/mg963/data/camels_us}
RAW=$ROOT/raw
OUT=$ROOT/CAMELS_US
HERE=$(cd "$(dirname "$0")" && pwd)

echo "== verifying checksums in $RAW"
cd "$RAW"
missing=0
while IFS=$'\t' read -r key md5 url; do
  if [ ! -f "$key" ]; then
    case "$key" in *modelOutput*|*.pdf|readme.txt) echo "  (optional, missing) $key"; continue;; esac
    echo "  MISSING $key"; missing=1; continue
  fi
  echo "$md5  $key" | md5sum -c --quiet || { echo "  BAD MD5 $key"; exit 1; }
done < files.tsv
[ "$missing" = 0 ] || { echo "Required files missing; see $RAW/download.sh"; exit 1; }

echo "== unpacking"
mkdir -p "$ROOT/unpacked"
if [ ! -d "$ROOT/unpacked/basin_dataset_public_v1p2" ]; then
  unzip -q basin_timeseries_v1p2_metForcing_obsFlow.zip -d "$ROOT/unpacked"
fi
if [ -f basin_timeseries_v1p2_modelOutput_daymet.zip ] && [ ! -e "$ROOT/unpacked/.modelOutput_daymet_done" ]; then
  unzip -q basin_timeseries_v1p2_modelOutput_daymet.zip -d "$ROOT/unpacked/modelOutput_daymet" \
    && touch "$ROOT/unpacked/.modelOutput_daymet_done"
fi

SRC=$(find "$ROOT/unpacked" -maxdepth 3 -type d -name basin_mean_forcing -path "*v1p2*" | head -1)
[ -n "$SRC" ] || { echo "basin_mean_forcing not found after unzip"; exit 1; }
SRC=$(dirname "$SRC")

echo "== arranging $OUT (symlinks)"
mkdir -p "$OUT/camels_attributes_v2.0"
ln -sfn "$SRC/basin_mean_forcing" "$OUT/basin_mean_forcing"
ln -sfn "$SRC/usgs_streamflow" "$OUT/usgs_streamflow"
for f in "$RAW"/camels_*.txt; do ln -sfn "$f" "$OUT/camels_attributes_v2.0/$(basename "$f")"; done

BASINS="${BASINS:-$HERE/../basins/531_basin_list.txt}"
nb=$(grep -c . "$BASINS")
echo "== sanity check against the basin list ($nb basins: $BASINS)"
nf=0; nq=0
while read -r b; do
  ls "$OUT"/basin_mean_forcing/daymet/*/"${b}"_lump_cida_forcing_leap.txt >/dev/null 2>&1 && nf=$((nf+1))
  ls "$OUT"/usgs_streamflow/*/"${b}"_streamflow_qc.txt >/dev/null 2>&1 && nq=$((nq+1))
done < "$BASINS"
echo "  daymet forcing files: $nf / $nb, streamflow files: $nq / $nb"
[ "$nf" = "$nb" ] && [ "$nq" = "$nb" ] || { echo "Some basins missing!"; exit 1; }
echo "Done. data_dir for configs: $OUT"
