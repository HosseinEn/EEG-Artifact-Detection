# run the py main.py for 10 times
# arange between -7 to 4 with step 0.5
SNR_VALUES=$(seq -7 0.5 -3)
for SNR_VALUE in $SNR_VALUES
do
  # write the SNR value to the config.yaml file under data > snr_db
  sed -i "s/snr_db: .*$/snr_db: $SNR_VALUE/" config.yaml
  for i in {1..10}
  do
      python3 main.py > /dev/null 2>&1
  done
done

