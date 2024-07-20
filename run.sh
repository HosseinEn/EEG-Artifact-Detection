LOWER_SNR=-2
UPPER_SNR=8
STEP=0.5
SNR_VALUES=$(seq $LOWER_SNR $STEP $UPPER_SNR)
for SNR_VALUE in $SNR_VALUES
do
    python3 main.py --snr_db "$SNR_VALUE" --mode train > /dev/null 2>&1
    for i in {1..3}
    do
        python3 main.py --snr_db "$SNR_VALUE" --mode test > /dev/null 2>&1
    done
done

#py -c "import pandas as pd; df = pd.read_csv('result.csv'); print(df.describe())"