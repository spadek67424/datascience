#!/bin/bash
for ((i=0;i<=10;i++))
do
	python3 classificationforadjust.py RDSN spambase.csv example_test.csv
done
