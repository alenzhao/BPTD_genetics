#!/bin/bash
#$ -cwd -S /bin/bash -j y
#$ -l mem=20G,time=10::
#$ -M sy2515@c2b2.columbia.edu -m bes
##$ -t 1-17

./main

