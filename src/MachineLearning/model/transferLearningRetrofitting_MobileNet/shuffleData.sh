#!/bin/sh
cWD="$(pwd)"
trainingDirectory="$(pwd)/trainingDataset/train"
valDirectory="$(pwd)/trainingDataset/test"
shuffle(){
#recursive
#for training to test movement
cd ${trainingDirectory}
folders=$(ls .)
for a in ${folders}; do
echo ${a} Type
totalFilesExists=$(find ${a} -type f | wc -l)
echo "total file ${totalFilesExists}"
randomizationFileLimit=$((${totalFilesExists} / 2))
echo "going to randomize ${randomizationFileLimit}"
echo "trainingMovement"
fileList=$(find ${a} -type f)
#fileSelect=$(find ${a} -type f | shuf -n 1)
echo "==="
count=0
while [ ${count} -lt ${randomizationFileLimit} ]; do
count=$(( ${count} + 1 ))
fileSelect=$( echo "${fileList}" | shuf -n 1 )
echo Randomly Selecting file from ${a} directory
echo "selected file to move ${fileSelect} to test ${valDirectory}" 
echo "training movement"
mv -v ${fileSelect} "${valDirectory}/${a}"
done
echo "==="
done

#for test to training
cd ${valDirectory}
folders=$(ls .)
for a in ${folders}; do
echo ${a} Type
totalFilesExists=$(find ${a} -type f | wc -l)
echo "total file ${totalFilesExists}"
randomizationFileLimit=$((${totalFilesExists} / 2))
echo "going to randomize ${randomizationFileLimit}"
fileList=$(find ${a} -type f)
echo "==="
count=0
while [ ${count} -lt ${randomizationFileLimit} ]; do
count=$(( ${count} + 1 ))
fileSelect=$( echo "${fileList}" | shuf -n 1 )
echo Randomly Selecting file from ${a} directory
echo "selected file to move ${fileSelect} to training ${trainingDirectory}"
echo "valMovement"
mv -v ${fileSelect} "${trainingDirectory}/${a}"
done 
echo "==="
done
}
iterationShuffle=100

iteration=0
echo "processing!"
while [ ${iteration} -lt ${iterationShuffle} ]; do
iteration=$(( ${iteration} + 1 ))
echo "Shuffling Data between validation data and trainig"
echo "Iteration ${iteration}"
shuffle > ${cWD}/logs.txt 2>&1
done
echo "done"
