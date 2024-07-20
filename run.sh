rm -rf output
LOWER_SNR=-2
UPPER_SNR=0
STEP=0.5
SNR_VALUES=$(seq $LOWER_SNR $STEP $UPPER_SNR)
for SNR_VALUE in $SNR_VALUES
do
    python3 main.py --snr_db "$SNR_VALUE" --mode train --no_plot
    for i in {1..1}
    do
        python3 main.py --snr_db "$SNR_VALUE" --mode test --no_plot
    done
done

#py -c "import pandas as pd; df = pd.read_csv('result.csv'); print(df.describe())"