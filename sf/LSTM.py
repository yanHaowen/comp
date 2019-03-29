import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences

alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_int = dict((c, i) for i, c in enumerate(alphabet))
int_to_char = dict((i, c) for i, c in enumerate(alphabet))
# prepare the dataset of input to output pairs encoded as integers
num_inputs = 1000
max_len = 5
dataX = []
dataY = []
for i in range(num_inputs):
    start = numpy.random.randint(len(alphabet)-2)
    end = numpy.random.randint(start, min(start+max_len,len(alphabet)-1))
    sequence_in = alphabet[start:end+1]
    sequence_out = alphabet[end + 1]
    dataX.append([char_to_int[char] for char in sequence_in])
    dataY.append(char_to_int[sequence_out])

X = pad_sequences(dataX, maxlen=5, dtype='float32')
X = numpy.reshape(X,(X.shape[0],5,1))
X = X / float(len(alphabet))
y=np_utils.to_categorical(dataY)

batch_size = 1
model = Sequential()
print X.shape[1]
model.add(LSTM(32))
model.add(Dense(y.shape[1],activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

#model.fit(X,y,epochs=100,batch_size=batch_size,verbose=2)
model.fit(X, y, epochs=100, batch_size=batch_size, verbose=2)


scores=model.evaluate(X,y,batch_size=batch_size,verbose=0)
model.reset_states()
print("Model Accuracy: %.2f%%" % (scores[1]*100))
for i in range(20):
    pattern_index = numpy.random.randint(len(dataX))
    pattern = dataX[pattern_index]
    x = pad_sequences([pattern], maxlen=5, dtype='float32')
    x = numpy.reshape(x, (1, 5, 1))
    x = x / float(len(alphabet))
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print seq_in, "->", result
