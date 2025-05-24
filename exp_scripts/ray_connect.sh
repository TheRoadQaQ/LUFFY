ray stop
#ray start --address=30.207.96.13:6379 --num-cpus=100
#ray start --address=30.207.96.23:6379 --num-cpus=100
export NUMEXPR_MAX_THREADS=100
export MASTER_ADDR=30.207.97.15
ray start --address=30.207.97.15:6379 --num-cpus=100
