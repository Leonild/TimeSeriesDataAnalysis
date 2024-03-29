#!/bin/bash

if [ $# -eq 0 ]
then
	echo Type the data path.
	exit 0
fi
for j in {100..400..100}
do
	for i in {2012..2022..1}
	do
		path=$1
		echo Agregating data "for $path/$j""km/$i.csv"
		python3 SpatialAgregation.py $path/$j"km" $i
	done
done
