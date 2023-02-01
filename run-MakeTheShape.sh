#!/bin/bash

if [ $1 -gt $2 ]
then
	echo The start number should be less than the final.
	echo Parameters are: "<First shape size!>" "<Last shape size!>" "<Number of shapes in this range!>"
	exit 0
fi
#calculating the increment number
jump=`expr $2 - $1`
aux=`expr $3 - 1`
jump=`expr $jump / $aux`
count=$1
aux=`expr $2 + 1`
while [ $aux -gt $count ]
do
	#making the areal unit shapefile to a specific length
	echo Generates the shape "$count"
	Rscript MakeTheShape.R $count ./USA-shape
	count=`expr $count + $jump`
done