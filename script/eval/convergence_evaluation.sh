#Daniels setup, please comment this Kevin when you want to use this script
SEQUENCE=( 2012_02_12 2012_04_29 2012_05_11 2012_06_15 2012_08_04 2012_10_28 2012_11_16 2012_12_01 )
started=$(date +%H_%M)
echo "Started evaluation $started"
echo "Target sequence: $SEQUENCE"
for seq in "${SEQUENCE[@]}"
do
./evaluate_sequence.sh $seq $started $1
done
