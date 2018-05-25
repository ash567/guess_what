import tensorflow as tf
from neural_toolbox import rnn, utils
from neural_toolbox.film_layer import FiLMResblock, film_layer

from generic.tf_factory.attention_factory import get_attention
# from generic.tf_utils.abstract_network import AbstractNetwork
from generic.tf_utils.abstract_network import ResnetModel
from generic.tf_factory.image_factory import get_image_features
import sys

# TODO DONE: Weight the loss of qgen and gusser. As of now, both are just added.
# class QGenGuesserNetworkLSTM(AbstractNetwork):
class QGenGuesserNetworkLSTM(ResnetModel):

	#TODO: add optional dropout inside and outside the lstm and images

	def __init__(self, config, num_words, policy_gradient, device='', reuse=False):
		# AbstractNetwork.__init__(self, "qgen_guesser", device=device)
		ResnetModel.__init__(self, "qgen_guesser", device=device)

		# Create the scope for this graph
		with tf.variable_scope(self.scope_name, reuse=reuse):

			# We set batch size to be none as the batch size for the validation set and train set are different
			# mini_batch_size = None
			mini_batch_size = config['batch_size']
			self.guesser_loss_weight = tf.constant(config["guesser_loss_weight"], dtype = tf.float32, name = "guesser_loss_weight")
			self.qgen_loss_weight = tf.constant(config["qgen_loss_weight"], dtype = tf.float32, name = "qgen_loss_weight")
			self.loss = 0    
			# *********************************************************
			# Placeholders specific for guesser and its processing
			# *********************************************************
			
			# Objects
			self.obj_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='obj_mask')
			self.obj_cats = tf.placeholder(tf.int32, [mini_batch_size, None], name='obj_cats')
			self.obj_spats = tf.placeholder(tf.float32, [mini_batch_size, None, config['spat_dim']], name='obj_spats')

			# Targets
			self.targets = tf.placeholder(tf.int32, [mini_batch_size], name="targets_index")

			self.object_cats_emb = utils.get_embedding(
				self.obj_cats,
				config['no_categories'] + 1,
				config['cat_emb_dim'],
				scope='cat_embedding')

			self.objects_input = tf.concat([self.object_cats_emb, self.obj_spats], axis=2)
			self.flat_objects_inp = tf.reshape(self.objects_input, [-1, config['cat_emb_dim'] + config['spat_dim']])

			with tf.variable_scope('obj_mlp'):
				h1 = utils.fully_connected(
					self.flat_objects_inp,
					n_out=config['obj_mlp_units'],
					activation='relu',
					scope='l1')
				h2 = utils.fully_connected(
					h1,
					n_out=config['no_hidden_final_mlp'],
					activation='relu',
					scope='l2')
			# print 
			# print 
			# print h2
			# TODO: Object Embeddings do not have image features right now
			obj_embs = tf.reshape(h2, [-1, tf.shape(self.obj_cats)[1], config['no_hidden_final_mlp']])


			# *********************************************************
			# Placeholders for Qgen and common placeholder for guesser and its processing
			# *********************************************************

			# Image
			self.images = tf.placeholder(tf.float32, [mini_batch_size] + config['image']["dim"], name='images')

			# Question
			self.dialogues = tf.placeholder(tf.int32, [mini_batch_size, None], name='dialogues')
			self.answer_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='answer_mask')  # 1 if keep and (1 q/a 1) for (START q/a STOP)
			self.padding_mask = tf.placeholder(tf.float32, [mini_batch_size, None], name='padding_mask')
			self.seq_length = tf.placeholder(tf.int32, [mini_batch_size], name='seq_length')

			# Rewards
			self.cum_rewards = tf.placeholder(tf.float32, shape=[mini_batch_size, None], name='cum_reward')


			# DECODER Hidden state (for beam search)
			zero_state = tf.zeros([1, config['num_lstm_units']])  # default LSTM state is a zero-vector
			zero_state = tf.tile(zero_state, [tf.shape(self.images)[0], 1])  # trick to do a dynamic size 0 tensors

			self.decoder_zero_state_c = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_c")
			self.decoder_zero_state_h = tf.placeholder_with_default(zero_state, [mini_batch_size, config['num_lstm_units']], name="state_h")
			decoder_initial_state = tf.contrib.rnn.LSTMStateTuple(c=self.decoder_zero_state_c, h=self.decoder_zero_state_h)

			# *******
			# Misc
			# *******
			
			self.is_training = tf.placeholder(tf.bool, name='is_training')
			self.greedy = tf.placeholder_with_default(False, shape=(), name="greedy") # use for graph
			self.samples = None

			# For each length of the answer, we are finding the next token

			# remove last token

			input_dialogues = self.dialogues[:, :-1]
			input_seq_length = self.seq_length - 1

			# remove first token(=start token)
			rewards = self.cum_rewards[:, 1:]
			target_words = self.dialogues[:, 1:]

			# to understand the padding:
			# input
			#   <start>  is   it   a    blue   <?>   <yes>   is   it  a    car  <?>   <no>   <stop_dialogue>
			# target
			#    is      it   a   blue   <?>    -      is    it   a   car  <?>   -   <stop_dialogue>  -

			# TODO:

			# 1. Include Film in guesser (Check if you are using film or cbn????)
			# Add finetune in the training (see the training and config file of clever)
				# Check the use of finetune (should we input pretrained model), normalize etc
				# See in the config file, if the attention has to be put inside image block
			# As of now, in the first part where we get image embedding, the we only flatten the image. Use RCNN or other method to get the image features.
			# Include attention on image given the dialog embedding in the guesser part
			# Include dropout om lstm (option for inside and outside) and image
			# Include attention on words given the image features
			# 2. Use tf.gather and use all the lstm states where there was yes or no (-) in the target and the stop dialog
			# 3. Make the code run
			# Check how does the is_training flag works
			
			# image processing
			with tf.variable_scope('image_feature') as img_scope:

				if len(config["image"]["dim"]) == 1:
					self.image_out = self.images
				else:
					# TODO: Create a different config for this attention
					# Putting images
					tf.summary.image("image", self.images)
					self.image_out = get_image_features(
						image=self.images, question = None,
						is_training=self.is_training,
						scope_name=img_scope.name,
						config=config['image'],
						att = False
					)
					
					image_pooling_size = [int((self.image_out).get_shape()[1]), int((self.image_out).get_shape()[2])]
					image_feature_depth = int((self.image_out).get_shape()[3])

					self.image_out = tf.layers.max_pooling2d(self.image_out,
																		image_pooling_size,
																		1,
																		padding='valid',
																		data_format='channels_last',
																		name='max_pooling_image_out')

					# self.filmed_picture_out = tf.layers.average_pooling2d(	self.filmed_picture_out,
					# 														final_pooling_size,
					# 														1,
					# 														padding='valid',
					# 														data_format='channels_last',
					# 														name='average_pooling_filmed_picture_out')

					# self.image_out = get_attention(self.images, None, config["image"]["attention"]) #TODO: improve by using the previous lstm state?
					# self.image_out = tf.contrib.layers.flatten(self.image_out)

				print self.image_out
				print
				print


				# Reduce the embedding size of the image
				with tf.variable_scope('image_embedding'):
					self.image_emb = utils.fully_connected(self.image_out,
														   config['image_embedding_size'])
					image_emb = tf.expand_dims(self.image_emb, 1)
					image_emb = tf.tile(image_emb, [1, tf.shape(input_dialogues)[1], 1])

			# Compute the question embedding

			input_words = utils.get_embedding(
				input_dialogues,
				n_words=num_words,
				n_dim=config['word_embedding_size'],
				scope="word_embedding")

			# concat word embedding and image embedding
			# TODO: Check the size (see if input_seq_length is increased or not)
			decoder_input = tf.concat([input_words, image_emb], axis=2, name="concat_full_embedding")

			# encode one word+image
			decoder_lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
					config['num_lstm_units'],
					layer_norm=False,
					dropout_keep_prob=1.0,
					reuse=reuse)

			# TODO: Since we have concatinated image, check if the input_seq_length should be increased by one
			# Decoding the states to generate questions
			self.decoder_output, self.decoder_state = tf.nn.dynamic_rnn(
				cell=decoder_lstm_cell,
				inputs=decoder_input,
				dtype=tf.float32,
				initial_state=decoder_initial_state,
				sequence_length=input_seq_length,
				scope="word_decoder")  # TODO: use multi-layer RNN

			max_sequence = tf.reduce_max(self.seq_length)

			# For the Guesser

			# Adding extra layers of LSTM
			# TODO: There are several default parameters in the fuction. Try using them
			# TODO: as of now, not using it.
			# TODO, as of now only using the hidden state, you may include the other state too
			last_states = self.decoder_state.h
			# last_states, _ = rnn.variable_length_LSTM_extension(
			#     self.decoder_output,
			#     self.decoder_state,
			#     num_hidden = config['num_lstm_units'],
			#     seq_length = input_seq_length
			#     )

			last_states = tf.reshape(last_states, [-1, config['num_lstm_units']])


			# TODO: Can be moved to utils  
			def masked_softmax(scores, mask):
				# subtract max for stability
				scores = scores - tf.tile(tf.reduce_max(scores, axis=(1,), keepdims=True), [1, tf.shape(scores)[1]])
				# compute padded softmax
				exp_scores = tf.exp(scores)
				exp_scores *= mask
				exp_sum_scores = tf.reduce_sum(exp_scores, axis=1, keepdims=True)
				return exp_scores / tf.tile(exp_sum_scores, [1, tf.shape(exp_scores)[1]])


			# compute the softmax for evaluation (on all the words on dialogue)
			with tf.variable_scope('decoder_output'):
				flat_decoder_output = tf.reshape(self.decoder_output, [-1, decoder_lstm_cell.output_size])
				flat_mlp_output = utils.fully_connected(flat_decoder_output, num_words)

				# retrieve the batch/dialogue format
				mlp_output = tf.reshape(flat_mlp_output, [tf.shape(self.seq_length)[0], max_sequence - 1, num_words])  # Ignore th STOP token

				self.softmax_output = tf.nn.softmax(mlp_output, name="softmax")
				self.argmax_output = tf.argmax(mlp_output, axis=2)
				self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mlp_output, labels=target_words)

			# compute the maximum likelihood loss for the dialogues (for valid words)
			with tf.variable_scope('ml_loss'):

				ml_loss = tf.identity(self.cross_entropy_loss)
				ml_loss *= self.answer_mask[:, 1:]  # remove answers (ignore the <stop> token)
				ml_loss *= self.padding_mask[:, 1:]  # remove padding (ignore the <start> token)

				# Count number of unmask elements
				count = tf.reduce_sum(self.padding_mask) - tf.reduce_sum(1 - self.answer_mask[:, :-1]) - 1  # no_unpad - no_qa - START token

				ml_loss = tf.reduce_sum(ml_loss, axis=1)  # reduce over dialogue dimension
				ml_loss = tf.reduce_sum(ml_loss, axis=0)  # reduce over minibatch dimension
				self.ml_loss = ml_loss / count  # Normalize

				self.qgen_loss = self.ml_loss
				self.loss += self.qgen_loss_weight * self.qgen_loss
				tf.summary.scalar("qgen_loss", self.qgen_loss)

			# TODO NOTE: IMP in config file, under the image section, set cbn to be true

			with tf.variable_scope('guesser_input') as scope:
				
				# Getting the CBN image features
				self.CBN_picture_out = get_image_features(
					image=self.images, question = last_states,
					is_training=self.is_training,
					scope_name=scope.name,
					config=config['image']
				)

				# FILMING the Features

				# self.filmed_picture_out = film_layer(ft=self.CBN_picture_out, context=last_states)
				# TODO: Make n a hyperparameter and add it to network parameters
				self.filmed_picture_out = self.CBN_picture_out
				n = 1
				for i in range(n):
					with tf.variable_scope('film_layer_' + str(i)):
						self.filmed_picture_out = FiLMResblock(features=self.filmed_picture_out, context=last_states, is_training=self.is_training).get()
						# self.filmed_picture_out_3 = FiLMResblock(features=self.filmed_picture_out_2, context=last_states, is_training=self.is_training).get()
						
				# self.filmed_picture_out = FiLMResblock(features=self.filmed_picture_out_3, context=last_states, is_training=self.is_training).get()                
				# self.filmed_picture_out = FiLMResblock(features=self.filmed_picture_out, context=last_states, is_training=self.is_training).get()                
				# self.filmed_picture_out = FiLMResblock(features=self.filmed_picture_out, context=last_states, is_training=self.is_training).get()                
				
				# self.filmed_picture_out = FiLMResblock(features=self.CBN_picture_out, context=last_states, is_training=self.is_training).get()

				# self.filmed_picture_out = film_layer(features=self.CBN_picture_out, context=last_states)


				# TODO: Doing a convolution over the feature maps (before the classifier)
				# Do a max pooling over the feature maps

				final_pooling_size = [int((self.filmed_picture_out).get_shape()[1]), int((self.filmed_picture_out).get_shape()[2])]
				final_feature_depth = int((self.filmed_picture_out).get_shape()[3])

				if str(config["pooling"]).lower() == 'max':
					self.filmed_picture_out = tf.layers.max_pooling2d(	self.filmed_picture_out,
																		final_pooling_size,
																		1,
																		padding='valid',
																		data_format='channels_last',
																		name='max_pooling_filmed_picture_out')

				elif str(config["pooling"]).lower() == 'avg':
					self.filmed_picture_out = tf.layers.average_pooling2d(	self.filmed_picture_out,
																			final_pooling_size,
																			1,
																			padding='valid',
																			data_format='channels_last',
																			name='average_pooling_filmed_picture_out')
				else:
					print "No Pooling defined"
					sys.exit()
				self.filmed_picture_out = tf.reshape(self.filmed_picture_out, [-1, final_feature_depth])
			
				# Combining filmed image and dialog features into one
				#####################

				activation_name = config["activation"]

				self.question_embedding = utils.fully_connected(last_states, config["no_question_mlp"], activation=activation_name, scope='question_mlp')
				self.picture_embedding = utils.fully_connected(self.filmed_picture_out, config["no_picture_mlp"], activation=activation_name, scope='picture_mlp')

				self.full_embedding = self.picture_embedding * self.question_embedding
				# self.full_embedding = tf.nn.dropout(full_embedding, dropout_keep)

				# self.guesser_out_0 = utils.fully_connected(self.full_embedding, config["no_hidden_prefinal_mlp"], scope='hidden_prefinal', activation=activation_name)
				self.guesser_out_0 = self.full_embedding

				# out = tf.nn.dropout(out, dropout_keep)
				
				# Since we are not having 
				# out = utils.fully_connected(out, no_answers, activation='linear', scope='layer_softmax')
				self.guesser_out = utils.fully_connected(self.guesser_out_0, config["no_hidden_final_mlp"], scope='hidden_final', activation=activation_name)
				self.guesser_out = tf.reshape(self.guesser_out, [-1, config["no_hidden_final_mlp"], 1])


			# TODO DONE: Add all these losses to tensorboard

			with tf.variable_scope('guesser_output'):
				# TODO: In paper they do dot product, but in code they do matmul !!
				scores = tf.matmul(obj_embs, self.guesser_out)
				scores = tf.reshape(scores, [-1, tf.shape(self.obj_cats)[1]])

				self.softmax = masked_softmax(scores, self.obj_mask)
				self.selected_object = tf.argmax(self.softmax, axis=1)

				self.guesser_error = tf.reduce_mean(utils.error(self.softmax, self.targets))
				self.guesser_loss = tf.reduce_mean(utils.cross_entropy(self.softmax, self.targets))
				self.loss += self.guesser_loss_weight * self.guesser_loss
 
				tf.summary.scalar("guesser loss", self.guesser_loss)

			# Compute policy gradient
			if policy_gradient:

				with tf.variable_scope('rl_baseline'):
					decoder_out = tf.stop_gradient(self.decoder_output)  # take the LSTM output (and stop the gradient!)

					flat_decoder_output = tf.reshape(decoder_out, [-1, decoder_lstm_cell.output_size])  #
					flat_h1 = utils.fully_connected(flat_decoder_output, n_out=100, activation='relu', scope='baseline_hidden')
					flat_baseline = utils.fully_connected(flat_h1, 1, activation='relu', scope='baseline_out')

					self.baseline = tf.reshape(flat_baseline, [tf.shape(self.seq_length)[0], max_sequence-1])
					self.baseline *= self.answer_mask[:, 1:]
					self.baseline *= self.padding_mask[:, 1:]


				with tf.variable_scope('policy_gradient_loss'):

					# Compute log_prob
					self.log_of_policy = tf.identity(self.cross_entropy_loss)
					self.log_of_policy *= self.answer_mask[:, 1:]  # remove answers (<=> predicted answer has maximum reward) (ignore the START token in the mask)
					# No need to use padding mask as the discounted_reward is already zero once the episode terminated

					# Policy gradient loss
					rewards *= self.answer_mask[:, 1:]
					self.score_function = tf.multiply(self.log_of_policy, rewards - self.baseline)  # score function

					self.baseline_loss = tf.reduce_sum(tf.square(rewards - self.baseline))

					self.policy_gradient_loss = tf.reduce_sum(self.score_function, axis=1)  # sum over the dialogue trajectory
					self.policy_gradient_loss = tf.reduce_mean(self.policy_gradient_loss, axis=0)  # reduce over minibatch dimension

					self.loss = self.policy_gradient_loss

			tf.summary.scalar("total network loss", self.loss)
			self.summary = tf.summary.merge_all()
			print('Model... build!')


	def get_loss(self):
		return self.loss

	# TODO: This should change
	def get_accuracy(self):
		return self.loss

	# TODO: This will also change.
	def build_sampling_graph(self, config, tokenizer, max_length=12):

		if self.samples is not None:
			return

		# define stopping conditions
		def stop_cond(states_c, states_h, tokens, seq_length, stop_indicator):

			has_unfinished_dialogue = tf.less(tf.shape(tf.where(stop_indicator))[0],tf.shape(stop_indicator)[0]) # TODO use "any" instead of checking shape
			has_not_reach_size_limit = tf.less(tf.reduce_max(seq_length), max_length)

			return tf.logical_and(has_unfinished_dialogue,has_not_reach_size_limit)


		# define one_step sampling
		with tf.variable_scope(self.scope_name):
			stop_token = tf.constant(tokenizer.stop_token)
			stop_dialogue_token = tf.constant(tokenizer.stop_dialogue)

		def step(prev_state_c, prev_state_h, tokens, seq_length, stop_indicator):
			input = tf.gather(tokens, tf.shape(tokens)[0] - 1)

			# Look for new finish dialogue
			is_stop_token = tf.equal(input, stop_token)
			is_stop_dialogue_token = tf.equal(input, stop_dialogue_token)
			is_stop = tf.logical_or(is_stop_token, is_stop_dialogue_token)
			stop_indicator = tf.logical_or(stop_indicator, is_stop)  # flag to false new finished dialogue

			# increment seq_length when the dialogue is not over
			seq_length = tf.where(stop_indicator, seq_length, tf.add(seq_length, 1))

			# compute the next words. TODO: factorize with qgen.. but how?!
			with tf.variable_scope(self.scope_name, reuse=True):
				word_emb = utils.get_embedding(
					input,
					n_words=tokenizer.no_words,
					n_dim=config['word_embedding_size'],
					scope="word_embedding",
					reuse=True)

				inp_emb = tf.concat([word_emb, self.image_emb], axis=1)
				with tf.variable_scope("word_decoder"):
					lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
						config['num_lstm_units'],
						layer_norm=False,
						dropout_keep_prob=1.0,
						reuse=True)

					state = tf.contrib.rnn.LSTMStateTuple(c=prev_state_c, h=prev_state_h)
					out, state = lstm_cell(inp_emb, state)

					# store/update the state when the dialogue is not finished (after sampling the <?> token)
					cond = tf.greater_equal(seq_length, tf.subtract(tf.reduce_max(seq_length), 1))
					state_c = tf.where(cond, state.c, prev_state_c)
					state_h = tf.where(cond, state.h, prev_state_h)


				with tf.variable_scope('decoder_output'):
					output = utils.fully_connected(state_h, tokenizer.no_words, reuse=True)

					sampled_tokens = tf.cond(self.greedy,
											 lambda: tf.argmax(output, 1),
											 lambda: tf.reshape(tf.multinomial(output, 1), [-1])
											 )
					sampled_tokens = tf.cast(sampled_tokens, tf.int32)

			tokens = tf.concat([tokens, tf.expand_dims(sampled_tokens, 0)], axis=0) # check axis!

			return state_c, state_h, tokens, seq_length, stop_indicator


		# initialialize sequences
		batch_size = tf.shape(self.seq_length)[0]
		seq_length = tf.fill([batch_size], 0)
		stop_indicator = tf.fill([batch_size], False)

		transpose_dialogue = tf.transpose(self.dialogues, perm=[1,0])

		self.samples = tf.while_loop(stop_cond, step, [self.decoder_zero_state_c,
													   self.decoder_zero_state_h,
													   transpose_dialogue,
													   seq_length,
													   stop_indicator],
									 shape_invariants=[self.decoder_zero_state_c.get_shape(),
													   self.decoder_zero_state_h.get_shape(),
													   tf.TensorShape([None, None]),
													   seq_length.get_shape(),
													   stop_indicator.get_shape()])

