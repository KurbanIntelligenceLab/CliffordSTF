#!/bin/bash
exec > >(tee run.log) 2>&1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
JOBDIR=$PWD
TV=$(python -c "import torch;print(torch.__version__.split('+')[0])")
export PYG="https://data.pyg.org/whl/torch-${TV}+cu124.html"

SRC=/root/work/goten_ms
rm -rf $SRC && mkdir -p $SRC && cd $SRC
cp $JOBDIR/cliffordstf_src.tgz $JOBDIR/run_train.py .
tar xzf cliffordstf_src.tgz
tar xzf $JOBDIR/md17_slim_bundle.tgz
export REPO=$SRC
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
SEEDS=${SEEDS:-"43 44 45"}
EPOCHS=${EPOCHS:-1000}
ESPAT=${ESPAT:-25}
MODEL_KIND=${MODEL_KIND:-controlled}

if [ "$MODEL_KIND" = "tuned" ]; then
  MODEL_ARGS="model.name=clifford_stf_full_10m model.cutoff=6.0 model.use_ema=true model.ema_decay=0.999"
  SCHED_ARGS="training.lr_scheduler.type=cosine training.lr_scheduler.T_max=$EPOCHS training.lr_scheduler.eta_min=1e-7"
  LR=5.0e-4; BS=8
else
  MODEL_ARGS="model.name=clifford_stf_full model.n_channels=80 model.n_hidden_output=96 model.cutoff=5.0 model.use_ema=true model.ema_decay=0.9"
  SCHED_ARGS="training.lr_scheduler.type=plateau training.lr_scheduler.patience=5 training.lr_scheduler.factor=0.8 training.lr_scheduler.min_lr=1e-7"
  LR=1.0e-4; BS=16
fi

echo "=== MULTI-SEED sweep: kind=$MODEL_KIND seeds=[$SEEDS] EPOCHS=$EPOCHS ES=$ESPAT conservative ==="
date

for SEED in $SEEDS; do
  echo ""; echo "================== SEED $SEED =================="; date
  CKPT_ROOT=/root/work/goten_ckpts_s${SEED}
  mkdir -p $CKPT_ROOT
  for MOL in $MOLS; do
    echo ""; echo "################ seed$SEED $MOL ################"; date
    OUT=$CKPT_ROOT/out_$MOL
    python run_train.py $MODEL_ARGS \
      seed=$SEED \
      model.direct_forces=false \
      dataset.name=md17 dataset.molecule=$MOL dataset.data_root=$DATA \
      dataset.split_dir=$SPLITS dataset.split_id=1 \
      dataset.max_train_samples=1000 dataset.val_frac=0.05 dataset.test_samples=1000 \
      dataset.energy_weight=1.0 dataset.force_weight=100.0 \
      training.batch_size=$BS training.monitor_metric=combined \
      training.epochs=$EPOCHS training.lr=$LR training.weight_decay=0.01 training.optimizer=adam \
      $SCHED_ARGS \
      training.early_stopping.patience=$ESPAT \
      output_root=$OUT 2>&1 | grep -E "Best|TRAIN_RC|Error|error|OutOfMemory|Traceback" | tail -8
    L=$(find $OUT -name logs.json | head -1)
    if [ -n "$L" ]; then
      cp "$L" $JOBDIR/logs_${MOL}_s${SEED}.json
      BEST=$(find $OUT -name "ckpt_best_val.pth" | head -1)
      LAST=$(find $OUT -name "ckpt_last.pth" | head -1)
      [ -n "$BEST" ] && cp "$BEST" $JOBDIR/ckpt_best_${MOL}_s${SEED}.pth
      [ -n "$LAST" ] && cp "$LAST" $JOBDIR/ckpt_last_${MOL}_s${SEED}.pth
      python -c "import json;d=json.load(open('$L'));t=d.get('test_metrics',{});vm=d.get('val_metrics',[]);print('seed$SEED $MOL best_ep=%s val_fcos=%.4f val_fmae=%.4f'%(d.get('best_epoch','?'),max((e['force_cos'] for e in vm),default=-1),min((e['force_mae'] for e in vm),default=-1)))"
    else echo "seed$SEED $MOL NO_LOGS"; fi
  done
  echo ""; echo "=== SEED $SEED COMPLETE ==="; date
done
echo ""; echo "=== MULTI-SEED SWEEP DONE ==="; date
ls -la $JOBDIR/logs_*_s*.json 2>/dev/null | tail -40
