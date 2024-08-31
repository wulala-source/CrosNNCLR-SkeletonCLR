import numpy as np
import tensorflow as tf
from net.gat import MGRN_S
import numpy as np
from test1 import gen_train_data
def SRL(J_in, J_bias_in,nb_nodes,batch_size=37646*2,time_step=50,hid_units = [3],Ps = [3, 1],ft_size = 3,residual = False):
	# # c=torch.tensor(J_in)
	nonlinearity = tf.nn.elu
	attn_drop = 0.0
	ffd_drop = 0.0
	is_train = True
	W_h = tf.Variable(tf.random_normal([3, hid_units[-1]]))
	b_h = tf.Variable(tf.zeros(shape=[hid_units[-1], ]))
	J_h = tf.reshape(J_in, [-1, ft_size])

	J_h = tf.matmul(J_h, W_h) + b_h
	J_h = tf.reshape(J_h, [batch_size*time_step, nb_nodes, hid_units[-1]])


	# cc = numpy.random.randint(2, size=(100, 25,3),dtype=float)
	J_seq_ftr = MGRN_S.inference(J_h, 0, 25, is_train,
							 attn_drop, ffd_drop,
							 bias_mat=J_bias_in,
							 hid_units=hid_units, n_heads=Ps,
							 residual=residual, activation=nonlinearity, r_pool=True)

	return J_seq_ftr


def CRL(s1, s2, s1_num, s2_num, hid_in):
	r_unorm = tf.matmul(s2, tf.transpose(s1, [0, 2, 1]))
	att_w = tf.nn.softmax(r_unorm)
	att_w = tf.expand_dims(att_w, axis=-1)
	s1 = tf.reshape(s1, [s1.shape[0], 1, s1.shape[1], hid_in])
	c_ftr = tf.reduce_sum(att_w * s1, axis=2)
	c_ftr = tf.reshape(c_ftr, [-1, hid_in])
	return r_unorm, c_ftr


def MGRN(J_in1, P_in1, B_in1, J_bias_in1, P_bias_in1, B_bias_in1):
	fusion_lambda=1
	batch_size=37646*2
	time_step=50
	ft_size=3
	hid_in=3
	hid_out=3
	nb_nodes=25
	J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, nb_nodes, ft_size))
	P_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 10, ft_size))
	B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 5, ft_size))
	# Interpolation
	J_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, nb_nodes))
	P_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 10, 10))
	B_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5))
	h_J_seq_ftr = SRL(J_in=J_in, J_bias_in=J_bias_in, nb_nodes=nb_nodes)
	h_P_seq_ftr = SRL(J_in=P_in, J_bias_in=P_bias_in, nb_nodes=10)
	h_B_seq_ftr = SRL(J_in=B_in, J_bias_in=B_bias_in, nb_nodes=5)

	h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_in])
	h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
	h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])

	W_cs_12 = tf.Variable(tf.random_normal([hid_in, hid_out]))
	W_cs_13 = tf.Variable(tf.random_normal([hid_in, hid_out]))

	W_self_1 = tf.Variable(tf.random_normal([hid_in, hid_out]))

	self_a_1, self_r_1 = CRL(h_J_seq_ftr, h_J_seq_ftr, nb_nodes, nb_nodes, hid_in)

	h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_in])
	h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
	h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])


	h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_in])
	h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
	h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])


	a_12, r_12 = CRL(h_P_seq_ftr, h_J_seq_ftr, 10, nb_nodes, hid_in)
	a_13, r_13 = CRL(h_B_seq_ftr, h_J_seq_ftr, 5, nb_nodes, hid_in)

	h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, hid_in])

	h_J_seq_ftr = h_J_seq_ftr + float(fusion_lambda) * (tf.matmul(self_r_1, W_self_1) + tf.matmul(r_12, W_cs_12) + tf.matmul(r_13, W_cs_13))
	h_J_seq_ftr = tf.reshape(h_J_seq_ftr, [-1, nb_nodes, hid_out])

	with tf.Session() as sess:
		sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
		h_J_seq_ftr1=sess.run(h_J_seq_ftr,
					  feed_dict={
								           J_in: J_in1,
								           P_in: P_in1,
								           B_in: B_in1,
								           J_bias_in: J_bias_in1,
								           P_bias_in: P_bias_in1,
								           B_bias_in: B_bias_in1,
	                                       })
		h_J_seq_ftr2=h_J_seq_ftr1.reshape(37646,50,2,25,3)
		h_J_seq_ftr1=np.transpose(h_J_seq_ftr2, (0,4, 1, 3, 2))
		sess.close()
	return h_J_seq_ftr1



path='/home/zxl/下载/ntu60/xview/train_position.npy'
data_numpy=np.load(path,encoding = "latin1")
J_in, P_in, B_in, J_bias_in, P_bias_in, B_bias_in=gen_train_data(data_numpy)
J_in=np.transpose(J_in,(0,2,4,3,1))
J_in=J_in.reshape(3764600,25,3)
P_in = np.transpose(P_in, (0,2,4,3,1))
P_in = P_in.reshape(3764600,10,3)
B_in = np.transpose(B_in, (0,2,4,3,1))
B_in = B_in.reshape(3764600,5,3)
# I_in = np.transpose(I_in, (1, 3, 2, 0))
# I_in = I_in.reshape(100, 49, 3)
data_numpy=MGRN(J_in, P_in, B_in,J_bias_in, P_bias_in, B_bias_in)
np.save('data_numpy',data_numpy)
print()