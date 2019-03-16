export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
config=./Config/config.cfg
device=cuda:0
log_name=log
# device ["cpu", "cuda:0", "cuda:1", ......]
nohup python -u main.py --config $config --device $device --train -p > $log_name 2>&1 &
tail -f $log_name

 


