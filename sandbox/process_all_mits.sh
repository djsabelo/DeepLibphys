#!/bin/bash
for i in $(seq 13 1 13)
do 
   gnome-terminal -e "python3 process_mit_ecgs.py $i"
   sleep 120
done
