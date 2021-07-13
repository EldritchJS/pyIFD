#!/bin/bash

INFILEPREFIX=$1
OUTFILEPREFIX=$2

declare -a Algorithms=("ADQ1" "ADQ2" "ADQ3" "BLK" "CAGI" "ELA" "GHO" "NOI1" "NOI2" "NOI4" "NOI5" )

for filename in `find ${INFILEPREFIX} -name *.jpg`
do
	echo "Processing ${filename}"
	OutFilenameBase=`basename $filename`
	OutFilenameNoExt=$(echo "$OutFilenameBase" | cut -f 1 -d '.')
	OutFiledir="${OUTFILEPREFIX}${OutFilenameNoExt}/"
	if [ ! -d $OutFiledir ]
	then
		mkdir -p $OutFiledir
		InFilename="../${filename}"
		for Algorithm in ${Algorithms[@]}
		do
			OutFilename="../${OutFiledir}${OutFilenameNoExt}_${Algorithm}.mat"
			python validate_algo.py $InFilename $Mat
			matlab -nodesktop -nodisplay -nojvm -nosplash -r "cd('$Algorithm'); ProcessAndSave('$InFilename','$OutFilename'); exit" 
		done
	else
		echo "Skipping ${OutFilenameBase} since output directory exists"
	fi
done


