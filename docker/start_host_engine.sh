#!/bin/sh

# start the DCGM service in background.
nohup nv-hostengine -n -b 0.0.0.0 -f log.txt & 

