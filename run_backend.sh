#!/bin/sh

SHELL=/bin/sh
PATH=/usr/local/sbin:/usr/local/bin:/sbin:/usr/sbin:/usr/bin

cd "$(dirname "$0")"

export PYTHONPATH=$PYTHONPATH:/home/jovyan/app/python-dotenv/src

python /home/jovyan/app/backend.py -i 94 
