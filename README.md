# EigenScape Tools [![DOI](https://zenodo.org/badge/79900362.svg)](https://zenodo.org/badge/latestdoi/79900362)

## by Marc Ciufo Green

Acoustic Scene Classification system designed for use with the EigenScape database. The main features of the module provided here are:

- Tools enabling easier manipulation and segmentation of the EigenScape database.
- Function for extraction of spatial features using [Directional Audio Coding (DirAC) techniques][1].
- MultiGMMClassifier object for classification using a bank of Gaussian Mixture Models.
- BOF_audio_classify function to classify audio clips using the ['Bag-of-Frames' method][2].
- Functions for easy plotting of ROC curves and confusion matrices.


#### Requirements:
- Python 3.6 or later
- Python modules:
  - [numpy](http://www.numpy.org/)
  - [scipy](https://www.scipy.org/)
  - [scikit-learn](http://scikit-learn.org/stable/)
  - [resampy](https://github.com/bmcfee/resampy)
  - [librosa](http://librosa.github.io/librosa/)
  - [pandas](http://pandas.pydata.org/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
  - [soundfile](https://pysoundfile.readthedocs.io/en/0.9.0/)
  - [progressbar](https://pypi.python.org/pypi/progressbar2)

Tested using Python 3.6.2 on Windows 10 and macOS 10.12.5


#### Usage examples
##### Creating test setup
```python
import eigenscape
eigenscape.datatools.create_test_setup('../EigenScape/')
```
By default, this will split all audio files in the 'EigenScape' directory into 30-second segments and shuffle each full recording into 4 folds for training and testing. These parameters can be overridden:

```python
eigenscape.datatools.create_test_setup('../EigenScape/', seg_length=20, n_folds=8)
```
This will split the audio into 20-second segments and shuffle the recordings into 8 folds (test audio clips will be from 1 single recording only). It is important to note that `seg_length` must be divisble by 600 seconds (10 minutes) and `n_folds` must be divisible by 8 (number of unique recordings per scene class in EigenScape).

Split audio files will be deposited in a folder named 'audio' and text files with information on the folds will be deposited in a folder named 'fold_info'.


##### Feature extraction
```python
data, indices, label_list = eigenscape.build_audio_featureset(
                              eigenscape.calculate_dirac,
                                dataset_directory='audio/')
```
This will use the DirAC feature extraction function built into the eigenscape module to calculate Azimuth, Elevation and Diffuseness estimates across 20 frequency bands by default, covering the frequency spectrum up to half the audio sampling frequency. The `hi_freq` and `n_bands` keyword arguments can be used to override these defaults.

FIR filters are used to split the audio into subbands. 2048-tap filters are used by default, but this can also be overridden using the `filt_taps` keyword argument. This could speed up the feature extraction but lead to lower precision.

`eigenscape.calculate_mfccs` can also be substituted in order to use librosa MFCC extraction in place of DirAC.

`build_audio_featureset` returns:
- A numpy array containing all the DirAC features for each frame of the audio in rows with class label numbers in the final column.
- A dictionary of indices indicating the audio segment from which features were extracted.
- A list of string labels for the scene classes present in the set.

##### Bag-of-Frames classification
```python
from sklearn.preprocessing import StandardScaler

X = data[:, :-1]
y = data[:, -1] # extract data vectors and class targets from array

scaler = StandardScaler() # set up scaler object

train_info = eigenscape.extract_info('fold_info/fold4_train.txt')
test_info = eigenscape.extract_info('fold_info/fold4_test.txt')
# read in file lists (4th fold here)

train_indices = eigenscape.vectorise_indices(train_info)
# make incremental vector of train data indices

X_train = X[train_indices]
y_train = y[train_indices]
# extract training data and labels from full arrays

classifier = eigenscape.MultiGMMClassifier() # set up multi GMM classifier

classifier.fit(scaler.fit_transform(X_train), y_train)
# train classifier on scaled training data and fit scaler to training data

y_test, y_score = eigenscape.BOF_audio_classify(
    classifier, scaler.transform(X), y, test_info, indices)
# classify entire audio clips (specified in test_info) by summing output from
# classifier object across all frames of the clip

```
The `BOF_audio_classify` function returns:
- `y_test` - an array with class labels for each full audio clip.
- `y_score` - an array containing probability scores from the classifier object.

The classifier object can be substituted for any scikit-learn object implementing the `decision_function` method.


##### Plotting results
```python
confmat = eigenscape.plot_confusion_matrix(y_test, y_score, label_list)
```
This will plot a confusion matrix based on the output from the classifier and return the confusion matrix as a numpy array.

`eigenscape.plot_roc` is also provided to plot ROC curves based on classifier output.

EigenScape database available at - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1012809.svg)](https://doi.org/10.5281/zenodo.1012809). 
[1]:http://www.aes.org/e-lib/browse.cfm?elib=14838
[2]:http://asa.scitation.org/doi/10.1121/1.2750160
