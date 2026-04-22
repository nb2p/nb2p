#!/usr/bin/env bash



export PYRO_SERIALIZERS_ACCEPTED=pickle
export PYRO_THREADPOOL_SIZE=128
export PYRO_SERIALIZER=pickle

python -m Pyro4.naming -n 0.0.0.0 &
sleep 0.1
python -m gensim.models.lsi_dispatcher &
sleep 0.1

for _ in {1..48}; do
  python -m gensim.models.lsi_worker &
  sleep 0.1
done
