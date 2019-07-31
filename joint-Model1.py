from __future__ import division, print_function, absolute_import
import statsmodels.api as sm
import gzip
import os
import re
import tarfile
import math
import random
import sys
import time
import logging
import numpy as np
import math
import rdkit

from tensorflow.python.platform import gfile
import tensorflow as tf
import keras
from keras.layers import BatchNormalization,Input,Reshape,Embedding,GRU,LSTM,Conv1D,Conv2D,LeakyReLU,MaxPooling1D,GlobalMaxPooling2D
from keras.layers import concatenate,Dropout,Dense,LeakyReLU,TimeDistributed,MaxPooling2D,add,Activation,SeparableConv2D
from keras import regularizers
from keras.optimizers import SGD,Adam
from keras.losses import mean_squared_error
from keras.models import Model
import keras.backend as K
from keras.activations import relu
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint,TensorBoard

from rdkit import Chem


random.seed(1234)
#### data and vocabulary

data_dir="./data/IC50"
vocab_size_protein=24
vocab_size_compound=15
vocab_size_ss=7
vocab_size_se=6
vocab_protein="vocab_protein"
vocab_ss="vocab_ss"
vocab_se="vocab_se"
#cid_dir = 'Kd/'
batch_size = 64
#image_dir = "Kd/"

GRU_size_prot=256
GRU_size_drug=64
GRU_size_ss=32
GRU_size_se=32
dev_perc=0.1

## Padding part
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(b"(\S)")
_WORD_SPLIT_2 = re.compile(b",")
_DIGIT_RE = re.compile(br"\d")
group_size = 50
num_group = 30
prot_max_size = 1500
comp_max_size = 75
## functions
def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    #if condition ==0:
    #    l = _WORD_SPLIT.split(space_separated_fragment)
    #    del l[0::2]
    #elif condition == 1:
    #    l = _WORD_SPLIT_2.split(space_separated_fragment)
    l = _WORD_SPLIT.split(space_separated_fragment)
    del l[0::2]
    words.extend(l)
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,normalize_digits=False):

  words = basic_tokenizer(sentence)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)



def data_to_token_ids(data_path, target_path, vocabulary_path,normalize_digits=False):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def read_data(source_path,MAX_size,group_size):
  data_set = []
  mycount=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()

        source_ids = [int(x) for x in source.split()]
        if len(source_ids) < MAX_size:
           pad = [PAD_ID] * (MAX_size - len(source_ids))
           #data_set.append(list(source_ids + pad)) #MK

           source_ids = list(source_ids + pad)
        elif len(source_ids) == MAX_size:
           #data_set.append(list(source_ids))

           #MK add alaki
           source_ids = list(source_ids)
        else:
           print("there is a data with length bigger than the max\n")
           print(len(source_ids))
        count=0
        temp=[]
        temp2=[]
        for x in source_ids:
          count=count+1
          if count < group_size+1:
            temp.append(x)
          if count == group_size+1:
            count=1
            temp2.append(temp)
            temp=[]
            temp.append(x)

        temp2.append(temp)
        data_set.append(temp2)

        mycount=mycount+1
        source = source_file.readline()
  return data_set

def read_graph(source_path,MAX_size):
  Vertex = []
  Adj = [] # Normalized adjacency matrix
  mycount=1
  PAD=0
  mydict={}
  max_size=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline().strip()
      counter = 0
      while source:
        mol = Chem.MolFromSmiles(source)
        atom_list = []
        for a in mol.GetAtoms():
            m = a.GetSymbol()
            if m not in mydict:
              mydict[m]=mycount
              mycount = mycount +1

            atom_list.append(mydict[m])

        if len(atom_list) > max_size:
           max_size = len(atom_list)


        if len(atom_list) < MAX_size:
           pad = [PAD] * (MAX_size - len(atom_list))
           atom_list = atom_list+pad


        vertex = np.array(atom_list, np.int32)
        Vertex.append(vertex)

        adja_mat = Chem.GetAdjacencyMatrix(mol)
        adj_temp = []
        for adja in adja_mat:
            if len(adja) < MAX_size:
               pad = [PAD]*(MAX_size - len(adja))
               adja = np.array(list(adja)+pad,np.int32)
            adj_temp.append(adja)
       cur_len = len(adj_temp)
        for i in range(MAX_size - cur_len):
            adja =np.array( [PAD]*MAX_size,np.int32)
            adj_temp.append(adja)

        adj_temp = adj_temp + np.eye(MAX_size) # A_hat = A + I
        deg = np.power(np.sum(adj_temp,axis=1),-0.5)
        deg_new = []
        for i in range(MAX_size):
            if deg[i]==1:
               deg_new.append(0)
            else:
               deg_new.append(deg[i])

        deg_new = np.array(deg_new)
        deg_diag = np.diag(deg_new)
        adj = np.matmul(deg_diag,adj_temp)
        adj = np.matmul(adj,deg_diag) # normalized
        Adj.append(adj)
        source = source_file.readline().strip()
  return Vertex,Adj,max_size

def prepare_data(data_dir, train_path, vocabulary_size,vocab,max_size,group_size):
  vocab_path = os.path.join(data_dir, vocab)
  train_ids_path = train_path + (".ids%d" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path)
  train_set = read_data(train_ids_path,max_size,group_size)

  return train_set

def read_labels(path):
    x = []
    f = open(path, "r")
    for line in f:
         if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
         else:
            x.append(float(line))

    return x

def  train_dev_split(train_protein,train_ss,train_se,train_compound_ver,train_compound_adj,train_IC50,dev_perc,comp_max_size,group_size,num_group,batch_size):
    num_whole= len(train_IC50)
    num_train = math.ceil(num_whole*(1-dev_perc)/batch_size)*batch_size
    num_dev = num_whole - num_train

    index_total = range(0,num_whole)
    index_dev = sorted(random.sample(index_total,num_dev))
    remain = list(set(index_total)^set(index_dev))
    index_train = sorted(random.sample(remain,num_train))

    compound_train_ver = [train_compound_ver[i] for i in index_train]
    compound_train_ver = np.reshape(compound_train_ver,[len(compound_train_ver),comp_max_size])
    compound_dev_ver = [train_compound_ver[i] for i in index_dev]
    compound_dev_ver = np.reshape(compound_dev_ver,[len(compound_dev_ver),comp_max_size])


    compound_train_adj = [train_compound_adj[i] for i in index_train]
    compound_train_adj = np.reshape(compound_train_adj,[len(compound_train_adj),comp_max_size,comp_max_size])
    compound_dev_adj = [train_compound_adj[i] for i in index_dev]
    compound_dev_adj = np.reshape(compound_dev_adj,[len(compound_dev_adj),comp_max_size,comp_max_size])


    IC50_train = [train_IC50[i] for i in index_train]
    IC50_train = np.reshape(IC50_train,[len(IC50_train),1])
    IC50_dev = [train_IC50[i] for i in index_dev]
    IC50_dev = np.reshape(IC50_dev,[len(IC50_dev),1])

    protein_train = [train_protein[i] for i in index_train]
    protein_train = np.reshape(protein_train,[len(protein_train),num_group,group_size])
    protein_dev = [train_protein[i] for i in index_dev]
    protein_dev = np.reshape(protein_dev,[len(protein_dev),num_group,group_size])

    ss_train = [train_ss[i] for i in index_train]
    ss_train = np.reshape(ss_train,[len(ss_train),num_group,group_size])
    ss_dev = [train_ss[i] for i in index_dev]
    ss_dev = np.reshape(ss_dev,[len(ss_dev),num_group,group_size])
    return compound_train_ver, compound_dev_ver,compound_train_adj, compound_dev_adj, IC50_train, IC50_dev, protein_train, protein_dev,ss_train,ss_dev,se_train,se_dev


## preparing data 
'''
train_protein = prepare_data(data_dir,data_dir+"/train_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
train_ss=prepare_data(data_dir,data_dir+"/train_ss",vocab_size_ss,vocab_ss,prot_max_size,group_size)
train_se=prepare_data(data_dir,data_dir+"/train_se",vocab_size_se,vocab_se,prot_max_size,group_size)
'''
print("the_lan")
#print(train_protein[0])
#train_protein=np.reshape(train_protein,[len(train_protein),num_group,group_size])
print("tuu")
#train_IC50 = read_labels(data_dir+"/train_ic50")
print("the_lan")
#train_compound_ver,train_compound_adj,train_compound_max = read_graph(data_dir+"/train_smile",comp_max_size)
print("the_lan")
'''
ER_protein = prepare_data(data_dir,data_dir+"/ER_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size) 
ER_IC50 = read_labels(data_dir+"/ER_kd")
ER_compound_ver,ER_compound_adj,ER_compound_max = read_graph(data_dir+"/ER_smile",comp_max_size)

GPCR_protein = prepare_data(data_dir,data_dir+"/GPCR_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size) 
GPCR_IC50 = read_labels(data_dir+"/GPCR_kd")
GPCR_compound_ver,GPCR_compound_adj,GPCR_compound_max = read_graph(data_dir+"/GPCR_smile",comp_max_size)

channel_protein = prepare_data(data_dir,data_dir+"/channel_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size) 
channel_IC50 = read_labels(data_dir+"/channel_kd")
channel_compound_ver,channel_compound_adj,channel_compound_max = read_graph(data_dir+"/channel_smile",comp_max_size)

kinase_protein = prepare_data(data_dir,data_dir+"/kinase_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size) 
kinase_IC50 = read_labels(data_dir+"/kinase_kd")
kinase_compound_ver,kinase_compound_adj,kinase_compound_max = read_graph(data_dir+"/kinase_smile",comp_max_size)
'''
test_protein = prepare_data(data_dir,data_dir+"/test_protein_seq",vocab_size_protein,vocab_protein,prot_max_size,group_size)
test_ss=prepare_data(data_dir,data_dir+"/test_ss",vocab_size_ss,vocab_ss,prot_max_size,group_size)
test_se=prepare_data(data_dir,data_dir+"/test_se",vocab_size_se,vocab_se,prot_max_size,group_size)
test_IC50 = read_labels(data_dir+"/test_ic50")
test_compound_ver,test_compound_adj,test_compound_max = read_graph(data_dir+"/test_smile",comp_max_size)

'''
print(len(train_IC50))
print(len(train_compound_ver))
print(len(train_compound_adj))
print(len(train_protein))
## separating train,dev, test data

compound_train_ver, compound_dev_ver,compound_train_adj, compound_dev_adj, IC50_train, IC50_dev, protein_train, protein_dev,ss_train,ss_dev,se_train,se_dev = train_dev_split(train_protein,train_ss,train_se,train_compound_ver,train_compound_adj,train_IC50,dev_perc,comp_max_size,group_size,num_group,batch_size)
'''
class Sep_attn(Layer):

    def __init__(self, output_dim,length, **kwargs):
        self.output_dim = output_dim
        self.length = length
        super(Sep_attn, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W',
                                      shape=(input_shape[2],self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)
        self.b = self.add_weight(name='b',
                                      shape=(input_shape[2],),
                                      initializer='random_uniform',
                                      trainable=True)
        self.U = self.add_weight(name='U',
                                      shape=(self.output_dim,1),
                                      initializer='random_uniform',
                                      trainable=True)

        super(Sep_attn, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        V = tf.tanh(tf.tensordot(x,self.W,axes=1)+self.b)
        V = Reshape((self.length,self.output_dim))(V)
        VU = tf.squeeze(tf.tensordot(V,self.U,axes=[[2],[0]]),axis=2)
        VU = Reshape((self.length,))(VU)
        alphas = tf.nn.softmax(VU,name='alphas')
        Attn = tf.expand_dims(tf.reduce_sum(x *tf.expand_dims(alphas,-1),1),2)
        return Attn
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class graph_layer(Layer):

    def __init__(self,output_dim,length, **kwargs):
        self.output_dim = output_dim
        self.length = length
        super(graph_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                                      shape=(self.output_dim,self.output_dim),
                                      initializer='random_uniform',
                                      trainable=True)

        super(graph_layer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        A = inputs[0]
        X = inputs[1]
        Z = tf.einsum('bij,bjk->bik', A, X)
        Y = tf.nn.relu(tf.einsum('bij,jk->bik', Z, self.W))
        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.length,self.output_dim)

prot_data = Input(shape=(num_group,group_size))
amino_embd = TimeDistributed(Embedding(input_dim = vocab_size_protein, output_dim = GRU_size_prot, input_length=group_size))(prot_data)
amino_lstm = TimeDistributed(GRU(units=GRU_size_prot,return_sequences=True))(amino_embd)
amino_lstm = TimeDistributed(Reshape((group_size,GRU_size_prot)))(amino_lstm)
prot_encoder = TimeDistributed(Sep_attn(output_dim=GRU_size_prot,length=group_size))(amino_lstm)

prot_lstm = GRU(units=GRU_size_prot,return_sequences=True)(prot_encoder)
prot_lstm = Reshape((num_group,GRU_size_prot))(prot_lstm)
prot_attention = Sep_attn(output_dim=GRU_size_prot,length=num_group)(prot_lstm)
prot_attention = Reshape((GRU_size_prot,1))(prot_attention)
conv_1 = Conv1D(filters=32,kernel_size=8,strides=4,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(prot_attention)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=4)(conv_1)
prot_final = Reshape((32*16,))(pool_1)

ss_data = Input(shape=(num_group,group_size))
ss_embd = TimeDistributed(Embedding(input_dim = vocab_size_ss, output_dim = GRU_size_ss, input_length=group_size))(ss_data)
ss_lstm = TimeDistributed(GRU(units=GRU_size_ss,return_sequences=True))(ss_embd)
ss_lstm = TimeDistributed(Reshape((group_size,GRU_size_ss)))(ss_lstm)
ss_encoder = TimeDistributed(Sep_attn(output_dim=GRU_size_ss,length=group_size))(ss_lstm)

ss1_lstm = GRU(units=GRU_size_ss,return_sequences=True)(ss_encoder)
ss1_lstm = Reshape((num_group,GRU_size_ss))(ss1_lstm)
ss1_attention = Sep_attn(output_dim=GRU_size_ss,length=num_group)(ss1_lstm)
ss1_attention = Reshape((GRU_size_ss,1))(ss1_attention)
conv_1 = Conv1D(filters=32,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(ss1_attention)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=2)(conv_1)
ss_final=Reshape((32*8,))(pool_1)

se_data = Input(shape=(num_group,group_size))
se_embd = TimeDistributed(Embedding(input_dim = vocab_size_se, output_dim = GRU_size_se, input_length=group_size))(se_data)
se_lstm = TimeDistributed(GRU(units=GRU_size_se,return_sequences=True))(se_embd)
se_lstm = TimeDistributed(Reshape((group_size,GRU_size_se)))(se_lstm)
se_encoder = TimeDistributed(Sep_attn(output_dim=GRU_size_se,length=group_size))(se_lstm)

se1_lstm = GRU(units=GRU_size_se,return_sequences=True)(se_encoder)
se1_lstm = Reshape((num_group,GRU_size_se))(se1_lstm)
se1_attention = Sep_attn(output_dim=GRU_size_se,length=num_group)(se1_lstm)
se1_attention = Reshape((GRU_size_se,1))(se1_attention)
conv_1 = Conv1D(filters=32,kernel_size=4,strides=2,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(se1_attention)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=2)(conv_1)
se_final=Reshape((32*8,))(pool_1)

## RNN for drug
drug_data_ver = Input(shape=(comp_max_size,))
drug_data_adj = Input(shape=(comp_max_size, comp_max_size))


drug_embd = Embedding(input_dim=vocab_size_compound, output_dim=GRU_size_drug)(drug_data_ver)

drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size)([drug_data_adj,drug_embd])
drug_embd = graph_layer(output_dim=GRU_size_drug,length=comp_max_size)([drug_data_adj,drug_embd])

drug_attention = Sep_attn(output_dim=GRU_size_drug,length=comp_max_size)(drug_embd)
drug_attention = Reshape((GRU_size_drug,1))(drug_attention)
conv_1 = Conv1D(filters=128,kernel_size=8,strides=4,padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=regularizers.l2(0.001))(drug_attention)
conv_1 = LeakyReLU(alpha=0.1)(conv_1)
pool_1 = MaxPooling1D(pool_size=4)(conv_1)
drug_final = Reshape((32*16,))(pool_1)

## merging
merging = concatenate([prot_final,ss_final,se_final,drug_final],axis=1)
fc_1 = Dense(units=800,kernel_initializer='glorot_uniform')(merging)
fc_1 = LeakyReLU(alpha=0.1)(fc_1)
drop_2 = Dropout(rate=0.8)(fc_1)
fc_2 = Dense(units=300,kernel_initializer='glorot_uniform')(drop_2)
fc_2 = LeakyReLU(alpha=0.1)(fc_2)
drop_3 = Dropout(rate=0.8)(fc_2)
linear = Dense(units=1,activation="linear",kernel_initializer='glorot_uniform')(drop_3)
model = Model(inputs=[prot_data,ss_data,se_data,drug_data_ver,drug_data_adj],outputs=[linear])

optimizer = Adam(0.0001)

filepath="hfinalweights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
tensorboard = TensorBoard(log_dir='./Graphfinal', histogram_freq=0, write_graph=True, write_images=True)
callbacks_list = [checkpoint,tensorboard]

model.load_weights('multifeature_model.h5')
'''

model.compile(loss=mean_squared_error,
              optimizer=optimizer)
# Training.
model.fit([protein_train,ss_train,se_train,compound_train_ver,compound_train_adj], IC50_train,
          batch_size=batch_size,
          epochs=46,
          verbose=1,
          validation_data=([protein_dev,ss_dev,se_dev,compound_dev_ver,compound_dev_adj], IC50_dev),
          callbacks=callbacks_list)
## saving
model.save('multifeature_model.h5')
'''
'''
print("error on dev")
size = 64
length_dev = len(protein_dev)
print(length_dev)
num_bins = math.ceil(length_dev/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([protein_dev[0:size],compound_dev_ver[0:size],compound_dev_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([protein_dev[(i*size):((i+1)*size)],compound_dev_ver[(i*size):((i+1)*size)],compound_dev_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([protein_dev[(i*size):length_dev],compound_dev_ver[(i*size):length_dev],compound_dev_adj[(i*size):length_dev]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
er=0
for i in range(length_dev):
  er += (y_pred[i]-IC50_dev[i])**2

mse = er/length_dev
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(IC50_dev)).fit()
print(results.summary())

print("error on ER")
ER_compound_adj = np.reshape(ER_compound_adj,[len(ER_compound_adj),comp_max_size,comp_max_size])
ER_compound_ver = np.reshape(ER_compound_ver,[len(ER_compound_ver),comp_max_size])
ER_protein = np.reshape(ER_protein,[len(ER_protein),num_group,group_size])

size = 64
length_ER = len(ER_protein)
print(length_ER)
num_bins = math.ceil(length_ER/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([ER_protein[0:size],ER_compound_ver[0:size],ER_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([ER_protein[(i*size):((i+1)*size)],ER_compound_ver[(i*size):((i+1)*size)],ER_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([ER_protein[length_ER-size:length_ER],ER_compound_ver[length_ER-size:length_ER],ER_compound_adj[length_ER-size:length_ER]])
          y_pred = np.concatenate((y_pred,temp[size-length_ER+(i*size):size]), axis=0)
er=0
for i in range(length_ER):
  er += (y_pred[i]-ER_IC50[i])**2

mse = er/length_ER
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(ER_IC50)).fit()
print(results.summary())

print("error on GPCR")
GPCR_compound_adj = np.reshape(GPCR_compound_adj,[len(GPCR_compound_adj),comp_max_size,comp_max_size])
GPCR_compound_ver = np.reshape(GPCR_compound_ver,[len(GPCR_compound_ver),comp_max_size])
GPCR_protein = np.reshape(GPCR_protein,[len(GPCR_protein),num_group,group_size])

size = 64
length_GPCR = len(GPCR_protein)
print(length_GPCR)
num_bins = math.ceil(length_GPCR/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([GPCR_protein[0:size],GPCR_compound_ver[0:size],GPCR_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([GPCR_protein[(i*size):((i+1)*size)],GPCR_compound_ver[(i*size):((i+1)*size)],GPCR_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([GPCR_protein[length_GPCR-size:length_GPCR],GPCR_compound_ver[length_GPCR-size:length_GPCR],GPCR_compound_adj[length_GPCR-size:length_GPCR]])
          y_pred = np.concatenate((y_pred,temp[size-length_GPCR+(i*size):size]), axis=0)

er=0
for i in range(length_GPCR):
  er += (y_pred[i]-GPCR_IC50[i])**2

mse = er/length_GPCR
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(GPCR_IC50)).fit()
print(results.summary())

print("error on kinase")
kinase_compound_adj = np.reshape(kinase_compound_adj,[len(kinase_compound_adj),comp_max_size,comp_max_size])
kinase_compound_ver = np.reshape(kinase_compound_ver,[len(kinase_compound_ver),comp_max_size])
kinase_protein = np.reshape(kinase_protein,[len(kinase_protein),num_group,group_size])

size = 64
length_kinase = len(kinase_protein)
print(length_kinase)
num_bins = math.ceil(length_kinase/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([kinase_protein[0:size],kinase_compound_ver[0:size],kinase_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([kinase_protein[(i*size):((i+1)*size)],kinase_compound_ver[(i*size):((i+1)*size)],kinase_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([kinase_protein[length_kinase-size:length_kinase],kinase_compound_ver[length_kinase-size:length_kinase],kinase_compound_adj[length_kinase-size:length_kinase]])
          y_pred = np.concatenate((y_pred,temp[size-length_kinase+(i*size):size]), axis=0)

er=0
for i in range(length_kinase):
  er += (y_pred[i]-kinase_IC50[i])**2

mse = er/length_kinase
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(kinase_IC50)).fit()
print(results.summary())

print("error on channel")
channel_compound_adj = np.reshape(channel_compound_adj,[len(channel_compound_adj),comp_max_size,comp_max_size])
channel_compound_ver = np.reshape(channel_compound_ver,[len(channel_compound_ver),comp_max_size])
channel_protein = np.reshape(channel_protein,[len(channel_protein),num_group,group_size])

size = 64
length_channel = len(channel_protein)
print(length_channel)
num_bins = math.ceil(length_channel/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([channel_protein[0:size],channel_compound_ver[0:size],channel_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([channel_protein[(i*size):((i+1)*size)],channel_compound_ver[(i*size):((i+1)*size)],channel_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([channel_protein[length_channel-size:length_channel],channel_compound_ver[length_channel-size:length_channel],channel_compound_adj[length_channel-size:length_channel]])
          y_pred = np.concatenate((y_pred,temp[size-length_channel+(i*size):size]), axis=0)

er=0
for i in range(length_channel):
  er += (y_pred[i]-channel_IC50[i])**2

mse = er/length_channel
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(channel_IC50)).fit()
print(results.summary())

print("error on train")
train_compound_adj = np.reshape(train_compound_adj,[len(train_compound_adj),comp_max_size,comp_max_size])
train_compound_ver = np.reshape(train_compound_ver,[len(train_compound_ver),comp_max_size])
train_protein = np.reshape(train_protein,[len(train_protein),num_group,group_size])

size = 64
length_train = len(train_protein)
print(length_train)
num_bins = math.ceil(length_train/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([train_protein[0:size],train_compound_ver[0:size],train_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([train_protein[(i*size):((i+1)*size)],train_compound_ver[(i*size):((i+1)*size)],train_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([train_protein[length_train-size:length_train],train_compound_ver[length_train-size:length_train],train_compound_adj[length_train-size:length_train]])
          y_pred = np.concatenate((y_pred,temp[size-length_train+(i*size):size]), axis=0)

er=0
for i in range(length_train):
  er += (y_pred[i]-train_IC50[i])**2

mse = er/length_train
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(train_IC50)).fit()
print(results.summary())
'''

print("error on test")
test_compound_adj = np.reshape(test_compound_adj,[len(test_compound_adj),comp_max_size,comp_max_size])
test_compound_ver = np.reshape(test_compound_ver,[len(test_compound_ver),comp_max_size])
test_protein = np.reshape(test_protein,[len(test_protein),num_group,group_size])
test_ss= np.reshape(test_ss,[len(test_ss),num_group,group_size])
test_se=np.reshape(test_se,[len(test_se),num_group,group_size])
size = 64
length_test = len(test_protein)
print(length_test)
num_bins = math.ceil(length_test/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test_protein[0:size],test_ss[0:size],test_se[0:size],test_compound_ver[0:size],test_compound_adj[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test_protein[(i*size):((i+1)*size)],test_ss[(i*size):((i+1)*size)],test_se[(i*size):((i+1)*size)],test_compound_ver[(i*size):((i+1)*size)],test_compound_adj[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test_protein[length_test-size:length_test],test_ss[length_test-size:length_test],test_se[length_test-size:length_test],test_compound_ver[length_test-size:length_test],test_compound_adj[length_test-size:length_test]])
          y_pred = np.concatenate((y_pred,temp[size-length_test+(i*size):size]), axis=0)

er=0
for i in range(length_test):
  er += (y_pred[i]-test_IC50[i])**2

mse = er/length_test
print(mse)
print(math.sqrt(mse))

results = sm.OLS(y_pred,sm.add_constant(test_IC50)).fit()
print(results.summary())

