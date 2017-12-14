# atiamML
Chemla - Latent representations for real-time synthesis space exploration

# Warning !
* Before each commit run unit tests 
* Name of datasets should NOT contain '_' characters (it is used )
* mb_size needs to be a divider of total dataset length (e.g. if len(dataset)=44800, then mb_size can be 100 or 448 etc...)

# Use of mainScript.py
###### Try:
* 'python mainScript.py --help'
###### Example: 
* to train a GAUSSIAN VAE on 'dataset.npz' of length 44800, data dim of 1024, z dim of 10, 1-'relu'-layer NN for Encoder and Decoder and save it in './savedVAE/' folder, the total command should be:
* 'python mainScript.py --encoderIOdims 1024 600 10 --decoderIOdims 10 600 1024 --encoderNL "relu6" --decoderNL "relu6" --type "gaussian" --dataset-path "../data/dataset.npz" --dataKey "images" --save-path "../data/savedVAE/" --mb-size 10 --epochs 10'
* The command can be reduced as it has default values (see '--help'):
* 'python mainScript.py --encoderIOdims 1024 600 10 --decoderIOdims 10 600 1024 --encoderNL "relu6" --decoderNL "relu6" --dataset-path "../data/dataset.npz" --save-path "../data/savedVAE/" '
* For a BERNOULLI VAE the equivalent command will be : 
* 'python mainScript.py --encoderIOdims 1024 600 10 --decoderIOdims 10 600 1024 --encoderNL "relu6" --decoderNL "relu6" "sigmoid" --type "bernoulli" --dataset-path "../data/dataset.npz" --save-path "../data/savedVAE/" '
###### Test: 
* an immediate working VAE training is on "./data/dummyDataset100Gaussian.npz". So try command :
* 'python mainScript.py --encoderIOdims 1024 600 10 --decoderIOdims 10 600 1024 --encoderNL "relu6" --decoderNL "relu6" '
* or Bernoulli equivalent :
* 'python mainScript.py --encoderIOdims 1024 600 10 --decoderIOdims 10 600 1024 --encoderNL "relu6" --decoderNL "relu6" "sigmoid" --type "bernoulli" '



# TODO
* Agreement for dataset structure (Alexis & Bavo) -> see ./src/dataset/ManageDataset.py
* Working with any mb_size
* Loading vae and restart training doesn't seem to work well
