# RevTransformerAttentionHAR
"**Transformer-based deep reverse attention network for multi-sensory human activity recognition**" - accepted in Engineering Applications of Artificial Intelligence, Elsevier.

Access to the journal article: Link coming soon


## Datasets Used:
The original credits for the dataset foes to the authors of the following repository: [https://github.com/RanaMostafaAbdElMohsen/Human_Activity_Recognition_using_Wearable_Sensors_Review_Challenges_Evaluation_Benchmark](https://github.com/RanaMostafaAbdElMohsen/Human_Activity_Recognition_using_Wearable_Sensors_Review_Challenges_Evaluation_Benchmark)
1. MHEALTH
2. USC-HAD
3. WHARF
4. UTD-MHAD1
5. UTD-MHAD2

Datasets can be found here: https://drive.google.com/drive/folders/13j488oaUwk_lufg9w9dvtExxw4wmOGVx 

## Instructions to run the code:

1. Download the repository and install the required packages:
```
pip3 install -r requirements.txt
```
2. The main.py file is sufficient for running the experiments. Run the code on terminal as follows:
```
python3 main.py --data_directory "data"
```
Available arguments:
- `--epochs`: Number of epochs of training. Default = 150
- `--folds`: Number of Folds of training. Default = 10
- `--batch_size`: Batch size for training. Default = 192
- `--learning_rate`: Initial Learning Rate. Default = 0.001
3. Edit the above parameters as per your requirement before running the code.
