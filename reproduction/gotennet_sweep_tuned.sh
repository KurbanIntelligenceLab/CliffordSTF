#!/bin/bash
exec > >(tee run.log) 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOBDIR=$PWD
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
export PYG="https://data.pyg.org/whl/torch-${TV}+cu124.html"
rm -rf /root/work/goten && mkdir -p /root/work/goten && cd /root/work/goten
cp $JOBDIR/cliffordstf_src.tgz $JOBDIR/run_train.py .
tar xzf cliffordstf_src.tgz
tar xzf $JOBDIR/md17_slim_bundle.tgz
export REPO=/root/work/goten
export DATA=$REPO/md17_slim
export SPLITS=$REPO/md17_slim/splits_slim

if [ ! -f /root/venv_base/bin/activate ]; then
  python -m venv --system-site-packages /root/venv_base; source /root/venv_base/bin/activate
  pip install -q --upgrade pip
  pip install -q torch_scatter torch_sparse torch_cluster -f "$PYG" 2>&1 | tail -1
  pip install -q torch_geometric omegaconf pytorch-lightning scikit-learn lmdb 2>&1 | tail -1
else source /root/venv_base/bin/activate; fi
cd $REPO; export PYTHONPATH="$REPO/src:$REPO"

MOLS=${MOLS:-"aspirin azobenzene benzene ethanol malonaldehyde naphthalene paracetamol salicylic toluene uracil"}
EPOCHS=${EPOCHS:-1000}
ESPAT=${ESPAT:-25}
CKPT_ROOT=/root/work/goten_ckpts
mkdir -p $CKPT_ROOT
echo "=== GotenNet-protocol sweep: EPOCHS=$EPOCHS earlystop=$ESPAT, energy-conservative, C=80/h=96 (~1.87M) ==="
for MOL in $MOLS; do
  echo ""; echo "################ $MOL ################"; date
  OUT=$CKPT_ROOT/out_$MOL
  python run_train.py model.name=clifford_stf_full_10m \
    model.direct_forces=false model.cutoff=6.0 model.use_ema=true model.ema_decay=0.999 \
    dataset.name=md17 dataset.molecule=$MOL dataset.data_root=$DATA \
    dataset.split_dir=$SPLITS dataset.split_id=1 \
    dataset.max_train_samples=1000 dataset.val_frac=0.05 dataset.test_samples=1000 \
    dataset.energy_weight=1.0 dataset.force_weight=100.0 \
    training.batch_size=8 training.monitor_metric=combined \
    training.epochs=$EPOCHS training.lr=5.0e-4 training.weight_decay=0.0 training.optimizer=adam \
    training.lr_scheduler.type=cosine training.lr_scheduler.T_max=$EPOCHS training.lr_scheduler.eta_min=1e-7 \
    training.early_stopping.patience=$ESPAT \
    output_root=$OUT 2>&1 | grep -E "Best|TRAIN_RC|Error|error|OutOfMemory|Traceback" | tail -8
  L=$(find $OUT -name logs.json | head -1)
  if [ -n "$L" ]; then
    cp "$L" $JOBDIR/logs_${MOL}.json
    MDIR=$(dirname "$L")/..
    BEST=$(find $OUT -name "ckpt_best_val.pth" | head -1)
    LAST=$(find $OUT -name "ckpt_last.pth" | head -1)
    [ -n "$BEST" ] && cp "$BEST" $JOBDIR/ckpt_best_${MOL}.pth
    [ -n "$LAST" ] && cp "$LAST" $JOBDIR/ckpt_last_${MOL}.pth
    python -c "import json;d=json.load(open('$L'));t=d.get('test_metrics',{});print('$MOL TEST energy_mae(kcal)=%.4f force_mae(kcal/A)=%.4f force_cos=%.4f (best_ep=%s)'%(t.get('energy_mae',-1),t.get('force_mae',-1),t.get('force_cos',-1),d.get('best_epoch','?')))"
  else echo "$MOL NO_LOGS"; fi
done
echo ""; echo "=== SWEEP DONE ==="; date; ls -la $JOBDIR/logs_*.json $JOBDIR/ckpt_*.pth 2>/dev/null
