#!/bin/bash

ImageFilePrefix=$1
MatFilePrefix=$2

#declare -a Algorithms=("ADQ1" "ADQ2" "ADQ3" "BLK" "CAGI" "ELA" "GHO" "NOI1" "NOI2" "NOI4" "NOI5" )
declare -a Algorithms=("NOI5" )

for ImageFilename in `find ${ImageFilePrefix} -name *.jpg | sort -n`
do
	echo "Processing ${ImageFilename}"
	MatFilenameBase=`basename $ImageFilename`
	MatFilenameNoExt=$(echo "$MatFilenameBase" | cut -f 1 -d '.')
	MatFiledir="${MatFilePrefix}/${MatFilenameNoExt}"
	for Algorithm in ${Algorithms[@]}
	do
		MatFilename="${MatFiledir}/${MatFilenameNoExt}_${Algorithm}.mat"
		python validate_algo.py $ImageFilename $MatFilename $Algorithm
	done
done


