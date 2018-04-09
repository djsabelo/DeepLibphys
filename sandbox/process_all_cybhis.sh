#!/bin/bash

for i in $(seq 0 1 7)
do 
	echo $i
	echo moment $m
   gnome-terminal -e "python3 train_cybhi_signals_v2.py 2 $i"
   sleep 120
done
