ray stop
#ray start --head --num-cpus=100 --node-ip-address=30.207.96.13 --port=6379
#ray start --head --num-cpus=100 --node-ip-address=30.207.96.23 --port=6379
export MASTER_ADDR=30.207.96.97
ray start --head --num-cpus=100 --node-ip-address=30.207.96.97 --port=6379
