#Dolapo's AWS Instance
from tensor2tensor.models import transformer

import numpy as np
import tensorflow as tf
import csv, random, time, string, re, io
import tensor2tensor as t2t

from tensorflow.python.client import timeline
from tensorflow.python import debug as tf_debug
from datetime import datetime
from collections import Counter

n_words = 10000

# load sentences and labels
with open('data-fs.txt', 'r') as f:
    sentences = f.readlines()
with open('labels-fs.txt', 'r') as f:
    labels = f.readlines()

#shuffle data
temp = list(zip(sentences, labels))
random.shuffle(temp)
sentences, labels = zip(*temp)

words = ' '.join(sentences).split()

#map words to integers
count = [['UNK', -1]]
count.extend(Counter(words).most_common(n_words - 1))
dictionary = dict()
for word, _ in count:
    dictionary[word] = len(dictionary)
data = list()
unk_count = 0
for word in words:
    if word in dictionary:
        index = dictionary[word]
    else:
        index = 0  # dictionary['UNK']
        unk_count += 1
    data.append(index)
count[0][1] = unk_count
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

#turn sentences into numbers - token ids
sentence_ints = []
for sentence in sentences:
    sentence_ints.append([dictionary[word] if word in dictionary else 0 \
                                            for word in sentence.split()])

#convert labels to numbers
labels = np.array([1 if _ == 'positive' else 0 for _ in labels])

#trim so that sentences are only of up to x words
features = np.zeros((len(sentence_ints), 500), dtype = int)
for _, row in enumerate(sentence_ints):
    features[_, -len(row):] = np.array(row)[:500]

#shuffle again
# temp = list(zip(features, labels))
# random.shuffle(temp)
# features, labels = zip(*temp)

_ = int(len(features) * .7)
train_x, val_x = features[:_], features[_:]
train_y, val_y = labels[:_], labels[_:]

_ = int(len(val_x) * .5)
val_x, test_x = val_x[:_], val_x[_:]
val_y, test_y = val_y[:_], val_y[_:]

#define hyperparameters
batch_size = 500 #should they be the same?
embed_size = 512
epochs = 10
learning_rate = .0005
lstm_sizes = [256, 128, 64]
hparams = transformer.transformer_base()
hparams.hidden_size = embed_size

#define generator batch function
def get_batches(x, y, batch_size = 100):
    n_batches = len(x) #batch_size
    #print(n_batches) #666167, size = 500
    x, y = x[:(n_batches * batch_size)], y[:(n_batches * batch_size)]
    for i in range(0, len(x), batch_size):
        if (len(x[i:(i + batch_size)]) == batch_size):
            yield x[i:(i + batch_size)], y[i:(i + batch_size)]

#create network
nn_graph = tf.Graph()
with nn_graph.as_default():
    _inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    _labels = tf.placeholder(tf.int32, [None, None], name = 'labels')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    embedding = tf.Variable(tf.random_uniform((n_words, embed_size), \
                                            minval = -1, maxval = 1))
    embed = tf.nn.embedding_lookup(embedding, _inputs) #embeddings
    
    model = transformer.Transformer(hparams)
    decoder = model.decode(embed, None, None, _inputs, hparams)
    print(decoder)
    exit() 
    #as opposed to None, use empty encoding vars 
    #   - encode an empty sequence = model encoder doing nothing
    
    outputs, _ = tf.contrib.seq2seq.dynamic_decode(decoder)
    logits = outputs.rnn_output

    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, \
                                                activation_fn = tf.sigmoid)
    cost = tf.losses.mean_squared_error(_labels, predictions)
    tf.summary.scalar("loss", cost)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), _labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    saver = tf.train.Saver()

#pass through rnn
with tf.Session(graph = nn_graph) as s:
    _time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer_val = tf.summary.FileWriter('./logs/{}/train'.format(_time))
    writer_train = tf.summary.FileWriter('./logs/{}/validation'.format(_time))

    s.run(tf.global_variables_initializer())
    #s = tf_debug.LocalCLIDebugWrapperSession(s)
    _iter = 1
    
    for e in range(epochs):
        write_op = tf.summary.merge_all()
        state = s.run(initial_state)
        #print(sum(1 for x in get_batches(train_x, train_y, batch_size))) #1332, size = 500
        
        for i, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):

            feed = {
                _inputs : x,
                _labels : y[:, None],
                keep_prob: 0.5,
                initial_state: state
            }

            loss, summary, acc, state, _ = s.run([cost, write_op, accuracy, \
                                    final_state, optimizer], feed_dict=feed)
            writer_train.add_summary(summary, _iter)

            if _iter % 5 == 0:
                print("Epoch: {}/{}".format(e, epochs),
                    "Iteration: {}".format(_iter),
                    "Train loss: {:.3f}".format(loss),
                    "Train acc: {:.3f}".format(acc))

            if _iter % 25 == 0:
                val_acc = []
                val_loss = []
                val_state = s.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {
                        _inputs : x,
                        _labels : y[:, None],
                        keep_prob: 1,
                        initial_state: val_state
                    }

                    summary, batch_loss, batch_acc, val_state = s.run([write_op, \
                                    cost, accuracy, final_state], feed_dict = feed)
                    val_acc.append(batch_acc)
                    val_loss.append(batch_loss)
                    writer_val.add_summary(summary, _iter)

                print(
                    "Validation Loss: {:.3f}".format(np.mean(val_loss)),
                    "Validation Accuracy: {:.3f}".format(np.mean(val_acc))
                    )

            if _iter % 100 == 0:
                checkpoint = "./model" + _time + ".ckpt"
                saver.save(s, checkpoint) #tell it to keep n chkpoints
            _iter += 1

# Test
test_acc = []
with tf.Session(graph = nn_graph) as s:
    saver.restore(s, tf.train.latest_checkpoint('.'))
    test_state = s.run(cell.zero_state(batch_size, tf.float32))

    for i, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {
            _inputs : x,
            _labels : y[:, None],
            keep_prob : 1,
            initial_state : test_state
        }

        batch_acc, test_state = s.run([accuracy, final_state], feed_dict = feed)
        test_acc.append(batch_acc)

    print("Testing Accuracy: {:.3f}".format(np.mean(test_acc)))