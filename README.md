# Keras / Tensorflow - predict next number
<img align="right" width="50%" src="https://www.beachsidetechnology.com/images/keras-number-seq-loss.png">
Use Keras and neural networks to predict the next number in a sequence of 1 dimensional numbers. This is a heavily commented neural network designed in Keras. It takes a series of 9 numbers and predicts the 10th. The model uses 3 layers of LSTM and is easy to modify. Number sequences are created in array format using a function for the purpose. The dataset is automatically created, so this is self-contained. 

# Tech Used
Should work on most setups with Keras installed and working.
- Tensorflow-gpu 2.5.0
- Keras 2.3.1
- Cuda 11.4

# Example Output 
We are asking the model to predict on this data: [[[  0.]
  [ 77.]
  [154.]
  [231.]
  [308.]
  [385.]
  [462.]
  [539.]
  [616.]]]
The model thinks this is the next number:  [[697.35236]]

# Neural Network Details
This network seemed to be about right for predictions. I ran 5000 epochs and got to low loss levels in less than 5 minutes of training. Tweaking this network will likely result in better performance and accuracy. I experimented with some layers being Dense, but found LSTM to have better accuracy with less time training.
- LSTM 64
- Dropout 0.1
- LSTM 32
- Dropout 0.1
- LSTM 32
- Dense 1 (linear)


# Troubleshooting
- Out of memory errors can be caused by leaving matplot graph open
