#coding=utf-8
#Author:Banbu
#Date:2019-06-21
#Email:liamao1995@163.com

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import re
import matplotlib.pyplot as plt
import jieba
from gensim.models import KeyedVectors

import warnings
import os

print('tf.__version__')

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(2)
tf.random.set_seed(2)
sentence_len = 300
batchsz = 128
words_num = 50000

# get the cn_model: including index matrix and embedding matrix
cn_model = KeyedVectors.load_word2vec_format('chinese_word_vectors/sgns.zhihu.bigram',
                                             binary=False)

embedding_dim = cn_model['a'].shape[0]


def embedding_matrix(words_num):
    # words_num: 50000
    # get embedding maxtrix: (50000, 300)
    embd_matrix = cn_model.vectors[:words_num, :]
    print('embedding matrix shape:', embd_matrix.shape)
    return embd_matrix


def word2index(word):
    # word to index
    # e.g: 我 => 1,  你 => 2
    try:
        return cn_model.vocab[word].index
    except KeyError:
        return 0


def index2word(index):
    if index != 0:
        return cn_model.index2word[index]

    return ' '


def load_model():
    embd_matrix = embedding_matrix(words_num)
    layer_list = [
        layers.Embedding(words_num,
                         embedding_dim,
                         weights=[embd_matrix],
                         input_length=sentence_len,
                         trainable=False),
        layers.Conv1D(filters=128, kernel_size=2, strides=2, padding='valid'),
        layers.MaxPool1D(pool_size=2, strides=2, padding='valid'),
        layers.Bidirectional(layers.LSTM(units=32, return_sequences=True)),
        layers.LSTM(units=16, return_sequences=False),
        layers.Dense(1, activation='sigmoid'),

    ]
    my_model = keras.Sequential(layer_list)

    optimizer = optimizers.Adam(lr=1e-3)
    my_model.compile(loss='binary_crossentropy',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    my_model.summary()
    return my_model


def load_data():
    # return 14k train, 2k test
    # x: list
    # y: numpy array

    # 1. data path
    # neg1_path = 'data/neg/neg_jiudian.txt'
    neg2_path = 'data/neg/neg_yuqing.txt'
    # pos1_path = 'data/pos/pos_jiudian.txt'
    pos2_path = 'data/pos/pos_yuqing.txt'

    # 2. read data and load to path list
    pos_samples, neg_samples = [], []

    #     for neg_sample in open(neg1_path, 'r', encoding='utf-8'):
    #         neg_samples.append(neg_sample)
    for neg_sample in open(neg2_path, 'r', encoding='utf-8'):
        neg_samples.append(neg_sample)
    #     for pos_sample in open(pos1_path, 'r', encoding='utf-8'):
    #         pos_samples.append(pos_sample)
    for pos_sample in open(pos2_path, 'r', encoding='utf-8'):
        pos_samples.append(pos_sample)

    # 3. remove the same sample
    neg_samples = list(set(neg_samples))
    pos_samples = list(set(pos_samples))
    print('length of original neg:', len(neg_samples))
    print('length of original pos:', len(pos_samples))
    # 4. train = 7k neg + 7k pos
    #    test  = 1k neg + 1k pos
    neg_train, neg_test = neg_samples[:6400], neg_samples[-500:]
    pos_train, pos_test = pos_samples[:6400], pos_samples[-500:]

    x_train_texts = neg_train + pos_train
    x_test_texts = neg_test + pos_test
    print(x_train_texts[:1])
    print(x_train_texts[-1:])
    print(x_test_texts[-1:])
    # 5. set the labels for train and test
    y_train = np.append(np.zeros(len(neg_train)),
                        np.ones(len(pos_train)),
                        axis=0)
    y_test = np.append(np.zeros(len(neg_test)),
                       np.ones(len(pos_test)),
                       axis=0)

    # texts => jieba.cut => word list => index list
    # x_train_texts => x_train
    # x_test_texts => x_test

    x_train = []
    x_test = []
    for text in x_train_texts:
        # use jieba to split words
        word_generator = jieba.cut(text)
        sentence = [word for word in word_generator]  # ['我'，’你‘]
        sentence = list(map(word2index, sentence))  # [1, 2]
        x_train.append(sentence)
    for text in x_test_texts:
        # use jieba to split words
        word_generator = jieba.cut(text)
        sentence = [word for word in word_generator]  # ['我'，’你‘]
        sentence = list(map(word2index, sentence))  # [1, 2]
        x_test.append(sentence)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)
    print('y_train average:', np.mean(y_train))
    print('y_test average:', np.mean(y_train))
    return x_train, y_train, x_test, y_test


def data2tensor():
    # data => padding => tensor
    x, y, x_test, y_test = load_data()
    print(y[-10:], y.max, )
    # pad_sequences: return numpy array data type.
    x = keras.preprocessing.sequence.pad_sequences(x,
                                                   maxlen=sentence_len,
                                                   padding='pre',
                                                   truncating='pre')
    x_test = keras.preprocessing.sequence.pad_sequences(x_test,
                                                        maxlen=sentence_len,
                                                        padding='pre',
                                                        truncating='pre')
    x[x >= words_num] = 0
    x_test[x_test >= words_num] = 0
    print('after padding, shape ', x.shape)
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.shuffle(10000).batch(batchsz, drop_remainder=True)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.batch(batchsz, drop_remainder=True)

    return db_train, db_test


def train():
    db_train, db_test = data2tensor()
    model = load_model()
    # define callback function
    ckp_path = 'data/weights.ckpt'
    sava_weights = keras.callbacks.ModelCheckpoint(ckp_path, monitor='val_loss',
                                                   verbose=0, save_best_only=True,
                                                   save_weights_only=True, mode='auto', period=1)

    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    lr_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     factor=0.1, min_lr=1e-5, patience=1,
                                                     verbose=1)
    callback_list = [sava_weights, earlystopping, lr_reduction]
    model.fit(db_train, epochs=50, validation_data=db_test, callbacks=callback_list)
    return model


def sentiment_predict(text, model):
    # 1. remove punctuation
    text = re.findall(r'[\u4e00-\u9fa5]', text)
    text = ''.join(text)
    # 2. use jieba to split word, jieba return a generator
    word_generator = jieba.cut(text)
    # 3. word generator => word list
    word_list = [word for word in word_generator]
    # 4. word list => index list
    index_list = list(map(word2index, word_list))
    # 5. padding to set length to 80
    x = keras.preprocessing.sequence.pad_sequences([index_list],
                                                   maxlen=sentence_len,
                                                   padding='pre',
                                                   truncating='pre')
    # 6. the num which more the words_num should be 0
    x[x > words_num] = 0
    # 7. predict

    result = model.predict(x=x)

    print(''.join(list(map(index2word, index_list))))
    print("score:%.2f" % result[0][0])
    if (result < 0.5):
        print("negative review!")
    else:
        print("postive review")
    print('-' * 30)
    return result


def test1(model):
    public_opinions_list = ['河南省教育厅:将尽快公布中原工学院教授涉论文抄袭初查意见',
                            '用时尚艺术向青春致敬!2019年中原工学院服装与服饰设计专业毕业生...',
                            '中原工学院成立前沿信息技术研究院',
                            '中原工学院宿舍失火 校园广播:禁止拍照 直接删了',
                            '中原工学院起诉学校餐厅承包商遭败诉 强制商户搬离',
                            '中原工学院图书馆发生火灾 万幸是夜间',
                            '中原工学院女大学生校园内被刀捅伤 究竟咋回事?',
                            '食堂里边的饭很难吃，吃完以后拉肚子',
                            '郑州大学第一附属医院内一男子坠楼身亡 警方介入调查',
                            '考生报考郑州大学却收到民办学校通知书 学生指责郑大招生欺骗',
                            '河南某高校某教授被检查出论文造假'
                            ]
    for txt in public_opinions_list:
        sentiment_predict(txt, model)


if __name__ == '__main__':
    model = train()
    test(model)
