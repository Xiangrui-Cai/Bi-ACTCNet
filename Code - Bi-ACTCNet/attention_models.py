import tensorflow as tf
from tensorflow.keras import layers


class MutualCrossAttention(layers.Layer):
	def __init__(self, dropout_rate, **kwargs):
		super(MutualCrossAttention, self).__init__(**kwargs)
		self.dropout = layers.Dropout(dropout_rate)

	def call(self, x1, x2):
		query = x1
		key = x2
		d = tf.cast(tf.shape(query)[-1], dtype=tf.float32)

		scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(d)
		attention_weights = tf.nn.softmax(scores, axis=-1)
		output_A = tf.matmul(self.dropout(attention_weights), x2)

		scores = tf.matmul(key, query, transpose_b=True) / tf.sqrt(d)
		attention_weights = tf.nn.softmax(scores, axis=-1)
		output_B = tf.matmul(self.dropout(attention_weights), x1)

		output = output_A + output_B

		return output

	def get_config(self):
		config = super(MutualCrossAttention, self).get_config()
		config.update({"dropout_rate": self.dropout.rate.numpy()})
		return config

