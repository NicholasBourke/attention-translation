Encoder-Decoder RNN with Attention based on "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau, Cho, Bengio; 2016).

Originally began as a PyTorch tutorial on pytorch.org but soon found it more fruitful to work directly from the original paper. Much of the data processing/loading code is still based on the tutorial content but the layer classes are written entirely from the paper.

Repository consists of a main branch and a testing branch. Main branch is the basic model designed to train over a number of epochs (saving parameters at each) and then immediately evaluating. Testing branch is where more realistic training methods that account for limited processing power were attempted. See discussion for examples.

There are four module files to support the main file. These are:

- layers.py (encoder and decoder classes)

- dataprocessing.py (Language class and functions used to pre-process the dataset)

- dataloading.py (functions to load data from .txt file and create dataloaders, ie. data functions called in training/evaluation process, unlke those in dataprocessing.py)

- training.py (training and evaluation functions called in main.py)




DISCUSSION

Maximum sentence length:

- It is necessary to set a maximum length T for both the source and target sentences. The longest sentence in the dataset is 56 words (incl. SOS and EOS), however the longest sentence in ~95% of sentence pairs has <15 words, so setting T=14 and filtering out pairs with longer sentences still retains the majority of the dataset. It is advantageous to do so to keep the training runtime reasonable (at least on a 2019 MacBook) as T impacts the dimensionality of sentence batch tensors and so large T can slow things down significantly.

- In the main branch we set T=56 and train on the entire dataset, however this takes way too long on my available hardware.

- In the testing branch the dataset is divided based on sentence length and a single training epoch is over each of these data subsets in turn (with T set appropriately). Since the bulk of the sentence pairs are short this drastically reduces training time, as the subsets with increasing max sentence length (and therefore increasing training-time-per-pair) contain fewer pairs.


Accuracy metric:

- Accuracy is judged on a per-word basis.

- Initially accuracy was unexpectedly high (~80%) but testing with actual translated output sentences showed this wasn't plausible. High accuracy results were coming from the inclusion of zero-padding in the accuracy calculations.

- Removal of any zeros shared between output and target (meaning output sentences that ran over the EOS into the padding were still judged innaccurate) from the accuracy calculation gave a more reasonable result, in line with what was observed in actual translation attempts.


Language class:

- Used to handle and process words and associated indices.

- French and English Language instances are pre-intialized and serialized/deserialized with Pickle.




RESULTS:

- Training in the testing branch is still time consuming so to date only the model has only been trained for two epochs of the full dataset.

- Accuracy so far is low, around 25% of words in output sentences match those in the source sentence.

- Model seems to have trouble producing the EOS token, often repeating the final word many times before doing so. Mid-sentence repetition is also an issue.

- Recently picked up a potential mistake in how the initial output word was created (randomly instead of with SOS - a holdover from a testing phase). Not sure if this explains the issue with the EOS token or the low accuracy in general, or if the model just requires more training.

- Many methods used in the paper (eg. regularization, adaptive learning rate) have not yet been implemented and could improve accuracy.

