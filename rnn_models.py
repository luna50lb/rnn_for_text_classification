import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D


class RNN2(Model):
	def __init__(self, input_dim, output_dim, emb_dim=300, hid_dim=100, embeddings=None, trainable=True):
		super(RNN2, self).__init__();
		self.model_name='rnn2'
		if embeddings is None:
			self.embedding = Embedding(input_dim=input_dim, output_dim=emb_dim, mask_zero=True, trainable=trainable, name='embedding')
		else:
			self.embedding = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], mask_zero=True,
			trainable=trainable, weights=[embeddings], name='embedding')
		print('...')
		self.rnn = SimpleRNN(hid_dim, name='rnn')
		self.fc = Dense(output_dim, activation='softmax')
		
	def call(self, inputs):
		v1=self.embedding(inputs)
		v2=self.rnn(v1);
		y=self.fc(v2)
		return y 


class LSTM2(Model):
	def __init__(self, input_dim, output_dim, emb_dim=300, hid_dim=100, embeddings=None, trainable=True):
		super(LSTM2, self).__init__();
		self.model_name='lstm2'
		if embeddings is None:
			self.embedding = Embedding(input_dim=input_dim, output_dim=emb_dim, mask_zero=True, trainable=trainable, name='embedding')
		else:
			self.embedding = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], mask_zero=True, trainable=trainable, 
			weights=[embeddings], name='embedding')
		print('...')
		self.lstm = LSTM(hid_dim, name='lstm')
		self.fc = Dense(output_dim, activation='softmax')
		
	def call(self, inputs):
		v1=self.embedding(inputs)
		v2=self.lstm(v1)
		y=self.fc(v2)
		return y


class LSTMCNN2(Model):

	def __init__(self, input_dim, output_dim, filters=250, kernel_size=3, emb_dim=300, hid_dim=100, embeddings=None):
		super(LSTMCNN2,self).__init__();
		self.model_name='lstm_cnn2';
		if embeddings is None:
			self.embedding = Embedding(input_dim=input_dim, output_dim=emb_dim, mask_zero=True, name='embedding')
		else:
			self.embedding = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], mask_zero=True, weights=[embeddings], name='embedding')
		self.lstm = LSTM(hid_dim, return_sequences=True, name='lstm')
		self.conv = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)
		self.pool = GlobalMaxPooling1D()
		self.fc1 = Dense(hid_dim)
		self.fc2 = Dense(output_dim, activation='softmax')
		
	def call(self, inputs):
		v1=self.embedding(inputs);
		v2=self.lstm(v1);
		v3=self.conv(v2);
		v4=self.pool(v3);
		v5=self.fc1(v4);
		y=self.fc2(v5);
		return y


class CNN2(Model):
	def __init__(self, input_dim, output_dim, filters=250, kernel_size=3, emb_dim=300, embeddings=None, trainable=True):
		super(CNN2, self).__init__();
		self.model_name='cnn2';
		if embeddings is None:
			self.embedding = Embedding(input_dim=input_dim, output_dim=emb_dim, trainable=trainable, name='embedding');
		else:
			self.embedding = Embedding(input_dim=embeddings.shape[0], output_dim=embeddings.shape[1], trainable=trainable, weights=[embeddings], name='embedding')
		self.conv = Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1)
		self.pool = GlobalMaxPooling1D()
		self.fc = Dense(output_dim, activation='softmax')
	
	def call(self, inputs):
		v1=self.embedding(inputs);
		v2=self.conv(v1);
		v3=self.pool(v2);
		y=self.fc(v3);	
		return y;



from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from tensorflow import keras

import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sys 


def import_dataset(v1:str):
	df_raw=pd.read_excel("data_20210319.xlsx",  header=0, sheet_name="dataset")
	df_raw['category_id']=df_raw.groupby(by=['category_name']).ngroup()
	return df_raw



def main():
	# Set hyper-parameters.
	batch_size = 5; # 128
	epochs = 100
	maxlen = 100; #300
	#model_path = 'models/model_{}.h5'
	num_words = 60; # 40000
	# num_label = 2
	learning_rate0=0.001
	#
	df_raw=import_dataset(v1='ww')
	#
	print('df_raw=\n', df_raw[ ['category_id', 'category_name'] ])
	x=df_raw['x'].values.tolist();
	y=df_raw['category_id'].values.tolist()
	num_label=max(y)+1;
	#print('type, shape', type(x), type(y), x.shape, y.shape )
	print('num_lable=', num_label)
	df_ref=df_raw.drop_duplicates(subset=['category_name', 'category_id'], keep='first').reset_index(drop=True)
	print('df_ref=\n', df_ref)
	
	#sys.exit()
	# pre-processing. 
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
	#
	#vocab = build_vocabulary(x_train, num_words)
	vocab0=tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token='<UNK>')
	vocab0.fit_on_texts(x_train)
	#
	x_train = vocab0.texts_to_sequences(x_train)
	x_test = vocab0.texts_to_sequences(x_test)
	print('A0 x_train=', x_train)
	x_train = pad_sequences(x_train, maxlen=maxlen, truncating='post')
	x_test = pad_sequences(x_test, maxlen=maxlen, truncating='post')
	print('A1 x_train=', x_train)
	y_train=np.array(y_train)
	print('x_train, y_train shape=', x_train.shape, y_train.shape )
	#
	#
	# 
	list_nns = [ 
	RNN2(num_words, num_label, embeddings=None ), 
	LSTM2(num_words, num_label, embeddings=None), 
		CNN2(num_words, num_label, embeddings=None),
		LSTMCNN2(num_words, num_label, embeddings=None), 
	]
	df_stacked=pd.DataFrame([]);
	df_train_hist=pd.DataFrame([]);
	for i, jnn in enumerate(list_nns):
		model=jnn;
		#if i>=4:
		#	model=jnn
		#else:
		#	model=jnn.build()
		print('i, model=',i,model)
		opt0=keras.optimizers.Adam(learning_rate=learning_rate0)
		#opt0=keras.optimizers.Adam()
		# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
		model.compile(optimizer=opt0, loss='sparse_categorical_crossentropy', metrics=['accuracy'] )
		# Preparing callbacks.
		# callbacks = [ EarlyStopping(patience=3), ModelCheckpoint(model_path.format(i), save_best_only=True) ]
		# 
		train_hist=model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
		#acc0=train_hist.history['acc']
		#range_epochs=np.arange(1, len(acc0)+1)
		#
		#print(train_hist.history['loss'])
		df_jhist=pd.DataFrame({'epoch':range(1, epochs+1), 'loss':train_hist.history['loss'], 'accuracy':train_hist.history['accuracy'] } )
		df_jhist['model_name']=jnn.model_name;
		df_train_hist=pd.concat([df_train_hist, df_jhist], axis=0)
		del(train_hist); del(df_jhist)
		#
		#
		yp0=model.predict(x_test)
		y_pred=np.argmax(yp0, -1)
		del(yp0)
		df_score=pd.DataFrame({'pred':y_pred, 'actual':y_test})
		print('pred, actual=', y_pred, y_test)
		df_score=df_score.merge(df_ref[['category_id', 'category_name']], left_on=['actual'], right_on=['category_id'], how='left', 
		validate='m:1', suffixes=('_l', '_r') )
		df_score['model_name']=jnn.model_name;
		print('df_score=\n', df_score[ ['pred', 'actual', 'category_name'] ] );
		df_stacked=pd.concat([df_stacked, df_score] ,axis=0);
		del(df_score)
		#print('precision: {:.4f}'.format(precision_score(y_test, y_pred, average=None)))
		#print('recall   : {:.4f}'.format(recall_score(y_test, y_pred, average=None)))
		#print('f1       : {:.4f}'.format(f1_score(y_test, y_pred, average=None)))

	df_stacked=df_stacked[['pred', 'actual', 'category_name', 'model_name'] ].reset_index(drop=True)
	def check_pred(dict_j):
		if dict_j['pred']==dict_j['actual']:
			ans0=1
		else:
			ans0=0;
		return ans0;		
	df_stacked['score']=df_stacked.apply(check_pred,axis=1)
	print('df_stacked=\n', df_stacked)
	print('score=\n', df_stacked.groupby(by=['model_name']).agg({'score':'mean' } ).reset_index() )
	fig0, (ax0, ax1)=plt.subplots(nrows=2)
	sns.lineplot(x='epoch',y='accuracy', data=df_train_hist, hue='model_name', ax=ax0)
	sns.lineplot(x='epoch',y='loss', data=df_train_hist, hue='model_name', ax=ax1)
	ax1.set_yscale("log");
	#ax0.plot(train_hist.history['accuracy'], label='Training acc')
	#ax1.plot(train_hist.history['loss'], label='Training loss')
	plt.tight_layout();
	plt.show()
	plt.close('all')

if __name__ == '__main__':
	main()

