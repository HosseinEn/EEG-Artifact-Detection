rm -rf output
LOWER_SNR=-2
UPPER_SNR=7
STEP=0.5
SNR_VALUES=$(seq $LOWER_SNR $STEP $UPPER_SNR)
for SNR_VALUE in $SNR_VALUES
do
    for i in {1..1}
    do
        # check if running on colab
        if [ -d "/content" ]; then
            python3 /content/SCEADNN/main.py --snr_db "$SNR_VALUE" --mode train --no_plot --datapath /content/SCEADNN/data --output /content/SCEADNN/output
        else
            python3 main.py --snr_db "$SNR_VALUE" --mode train --no_plot
        fi
    done
done

#py -c "import pandas as pd; df = pd.read_csv('result.csv'); print(df.describe())"