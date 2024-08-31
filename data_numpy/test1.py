import torch
import numpy as np
import pickle as pkl
# import networkx as nx
import scipy.sparse as sp
import copy
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

"""
 Generate training data for each dataset.
"""

def gen_train_data(data_numpy):
	nb_nodes=25
	nhood=1
	X_train_J = data_numpy
	X_train_P = reduce2part(X_train_J, nb_nodes)
	X_train_B = reduce2body(X_train_J, nb_nodes)

	def generate_denser_adj(adj):
		adj_temp = copy.deepcopy(adj).tolist()
		node_num = len(adj_temp)
		# if global_att:
		# 	new_adj = np.ones([node_num * 2 - 1, node_num * 2 - 1])
		# 	return new_adj
		new_adj = np.zeros([node_num * 2 - 1, node_num * 2 - 1])
		cnt = node_num
		for i in range(node_num):
			for j in range(node_num):
				if adj_temp[i][j] == 1:
					new_adj[i, cnt] = new_adj[cnt, i] = new_adj[j, cnt] = new_adj[cnt, j] = 1
					adj_temp[i][j] = adj_temp[j][i] = 0
					# print(i, j, cnt)
					cnt += 1
		for i in range(node_num):
			for j in range(node_num):
				if adj_temp[i][j] == 1:
					assert new_adj[i, j] == new_adj[j, i] == 0
		# print(cnt)
		# np.save('interp_graph_adj.npy', new_adj)
		return new_adj

	import scipy.sparse

	j_pair_1 = np.array([3, 2, 20, 8, 8, 9, 10, 9, 11, 10, 4, 20, 4, 5, 5, 6, 6, 7, 1, 20, 1, 0, 16, 0,
					12, 0, 16, 17, 12, 13, 17, 18, 19, 18, 13, 14, 14, 15, 2, 20, 11, 23, 10, 24, 7, 21, 6, 22])
	j_pair_2 = np.array([2, 3, 8, 20, 9, 8, 9, 10, 10, 11, 20, 4, 5, 4, 6, 5, 7, 6, 20, 1, 0, 1, 0, 16,
					0, 12, 17, 16, 13, 12, 18, 17, 18, 19, 14, 13, 15, 14, 20, 2, 23, 11, 24, 10, 21, 7, 22, 6])
	con_matrix = np.ones([48])
	adj_joint = scipy.sparse.coo_matrix((con_matrix, (j_pair_1, j_pair_2)), shape=(nb_nodes, nb_nodes)).toarray()
	# adj_interp = generate_denser_adj(adj_joint)

	# Part-Level adjacent matrix
	p_pair_1 = np.array([5, 6, 5, 8, 6, 7, 8, 9, 5, 4, 4, 2, 4, 0, 2, 3, 1, 0])
	p_pair_2 = np.array([6, 5, 8, 5, 7, 6, 9, 8, 4, 5, 2, 4, 0, 4, 3, 2, 0, 1])
	con_matrix = np.ones([18])
	adj_part = scipy.sparse.coo_matrix((con_matrix, (p_pair_1, p_pair_2)), shape=(10, 10)).toarray()

	# Body-Level adjacent matrix
	b_pair_1 = np.array([2, 3, 2, 4, 2, 1, 2, 0])
	b_pair_2 = np.array([3, 2, 4, 2, 1, 2, 0, 2])
	con_matrix = np.ones([8])
	adj_body = scipy.sparse.coo_matrix((con_matrix, (b_pair_1, b_pair_2)), shape=(5, 5)).toarray()

	# X_train_In = generate_denser_graph_data(X_train_J, adj_joint, nb_nodes)

	adj_joint = adj_joint[np.newaxis]
	biases_joint = adj_to_bias(adj_joint, [nb_nodes], nhood=nhood)

	adj_part = adj_part[np.newaxis]
	biases_part = adj_to_bias(adj_part, [10], nhood=1)

	adj_body = adj_body[np.newaxis]
	biases_body = adj_to_bias(adj_body, [5], nhood=1)

	# adj_interp = adj_interp[np.newaxis]
	# biases_interp = adj_to_bias(adj_interp, [nb_nodes*2-1], nhood=1)

	# return X_train_J, X_train_P, X_train_B, X_train_In, adj_joint, biases_joint, adj_part, biases_part, adj_body, biases_body, adj_interp, biases_interp
	return X_train_J, X_train_P, X_train_B, biases_joint, biases_part, biases_body


"""
 Generate part-level  skeleton graphs.
"""
def reduce2part(X, joint_num=25):
    left_leg_up = [16, 17]
    left_leg_down = [18, 19]
    right_leg_up = [12, 13]
    right_leg_down = [14, 15]
    torso = [0, 1]
    head = [2, 3, 20]
    left_arm_up = [8, 9]
    left_arm_down = [10, 11, 23, 24]
    right_arm_up = [4, 5]
    right_arm_down = [6, 7, 21, 22]
    # X = X.cpu().numpy()
    x_torso = np.mean(X[:, :, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftlegup = np.mean(X[:, :, :, left_leg_up, :], axis=3)
    x_leftlegdown = np.mean(X[:, :, :, left_leg_down, :], axis=3)
    x_rightlegup = np.mean(X[:, :, :, right_leg_up, :], axis=3)
    x_rightlegdown = np.mean(X[:, :, :, right_leg_down, :], axis=3)
    x_head = np.mean(X[:, :, :, head, :], axis=3)
    x_leftarmup = np.mean(X[:, :, :, left_arm_up, :], axis=3)
    x_leftarmdown = np.mean(X[:, :, :, left_arm_down, :], axis=3)
    x_rightarmup = np.mean(X[:, :, :, right_arm_up, :], axis=3)
    x_rightarmdown = np.mean(X[:, :, :, right_arm_down, :], axis=3)
    X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup, x_rightlegdown, x_torso, x_head, x_leftarmup,
                             x_leftarmdown, x_rightarmup, x_rightarmdown), axis=-1) \
                            .reshape([X.shape[0], X.shape[1],X.shape[2], 10, 2])
    # X_part = np.concatenate((x_torso, x_head, x_rightarmup, x_rightarmdown, x_leftarmup, x_leftarmdown,
    #                          x_rightlegup, x_rightlegdown, x_leftlegup, x_leftlegdown), axis=-1) \
    #                         .reshape([X.shape[0], X.shape[1], X.shape[2], 10, 2])
    # X_part=torch.tensor(X_part).cuda()
    return X_part

"""
 Generate body-level  skeleton graphs.
"""
def reduce2body(X, joint_num=25):
    left_leg = [16, 17, 18, 19]
    right_leg = [12, 13, 14, 15]
    torso = [0, 1, 2, 3, 20]
    left_arm = [8, 9, 10, 11, 23, 24]
    right_arm = [4, 5, 6, 7, 21, 22]
    x_torso = np.mean(X[:,:, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftleg = np.mean(X[:,:, :, left_leg, :], axis=3)
    x_rightleg = np.mean(X[:,:, :, right_leg, :], axis=3)
    x_leftarm = np.mean(X[:,:, :, left_arm, :], axis=3)
    x_rightarm = np.mean(X[:,:, :, right_arm, :], axis=3)
    X_body = np.concatenate((x_leftleg, x_rightleg, x_torso, x_leftarm, x_rightarm), axis=-1)\
                            .reshape([X.shape[0], X.shape[1],X.shape[2], 5, 2])
    return X_body


"""
 Linear interpolation
"""
def interpolation(X, joint_num=25):
    left_leg_up = [16, 17]
    left_leg_down = [18, 19]
    right_leg_up = [12, 13]
    right_leg_down = [14, 15]
    torso = [0, 1]
    head_1 = [2, 3]
    head_2 = [2, 20]
    left_arm_up = [8, 9]
    left_arm_down_1 = [10, 11]
    left_arm_down_2 = [11, 24]
    left_arm_down_3 = [24, 23]
    right_arm_up = [4, 5]
    right_arm_down_1 = [6, 7]
    right_arm_down_2 = [7, 22]
    right_arm_down_3 = [22, 21]
    shoulder_1 = [8, 20]
    shoulder_2 = [4, 20]
    elbow_1 = [9, 10]
    elbow_2 = [5, 6]
    spine_mm = [20, 1]
    hip_1 = [0, 16]
    hip_2 = [0, 12]
    knee_1 = [17, 18]
    knee_2 = [13, 14]
    x_torso = np.mean(X[:,:, :, torso, :], axis=3)  # [N * T, V=1]
    x_leftlegup = np.mean(X[:,:, :, left_leg_up, :], axis=3)
    x_leftlegdown = np.mean(X[:,:, :, left_leg_down, :], axis=3)
    x_rightlegup = np.mean(X[:,:, :, right_leg_up, :], axis=3)
    x_rightlegdown = np.mean(X[:,:, :, right_leg_down, :], axis=3)
    x_head_1 = np.mean(X[:,:, :, head_1, :], axis=3)
    x_head_2 = np.mean(X[:,:, :, head_2, :], axis=3)
    x_leftarmup = np.mean(X[:,:, :, left_arm_up, :], axis=3)
    x_leftarmdown_1 = np.mean(X[:,:, :, left_arm_down_1, :], axis=3)
    x_leftarmdown_2 = np.mean(X[:,:, :, left_arm_down_2, :], axis=3)
    x_leftarmdown_3 = np.mean(X[:,:, :, left_arm_down_3, :], axis=3)
    x_rightarmup = np.mean(X[:,:, :, right_arm_up, :], axis=3)
    x_rightarmdown_1 = np.mean(X[:,:, :, right_arm_down_1, :], axis=3)
    x_rightarmdown_2 = np.mean(X[:,:, :, right_arm_down_2, :], axis=3)
    x_rightarmdown_3 = np.mean(X[:,:, :, right_arm_down_3, :], axis=3)
    shoulder_1 = np.mean(X[:,:, :, shoulder_1, :], axis=3)
    shoulder_2 = np.mean(X[:,:, :, shoulder_2, :], axis=3)
    elbow_1 = np.mean(X[:,:, :, elbow_1, :], axis=3)
    elbow_2 = np.mean(X[:,:, :, elbow_2, :], axis=3)
    spine_mm = np.mean(X[:,:, :, spine_mm, :], axis=3)
    hip_1 = np.mean(X[:,:, :, hip_1, :], axis=3)
    hip_2 = np.mean(X[:,:, :, hip_2, :], axis=3)
    knee_1 = np.mean(X[:,:, :, knee_1, :], axis=3)
    knee_2 = np.mean(X[:,:, :, knee_2, :], axis=3)
    X_part = np.concatenate((x_leftlegup, x_leftlegdown, x_rightlegup,
                             x_rightlegdown, x_torso, x_head_1, x_head_2, x_leftarmup,
                             x_leftarmdown_1, x_leftarmdown_2, x_leftarmdown_3,
                             x_rightarmup, x_rightarmdown_1, x_rightarmdown_2, x_rightarmdown_3,
                             shoulder_1, shoulder_2, elbow_1, elbow_2, spine_mm,
                             hip_1, hip_2, knee_1, knee_2), axis=-1) \
        .reshape([X.shape[0], X.shape[1],X.shape[2], 24, 2])
    # 25+24
    X_interp = np.concatenate((X, X_part), axis=-2)
    return X_interp

def generate_denser_graph_data(X, adj, joint_num=25):#获得密集的图数据
	adj_temp = copy.deepcopy(adj)
	adj_temp = adj_temp.tolist()
	node_num = len(adj_temp)
	cnt = node_num
	for i in range(node_num):
		for j in range(node_num):
			if adj_temp[i][j] == 1:
				adj_temp[i][j] = adj_temp[j][i] = 0
				new_node = np.mean(X[:,:, :, [i, j], :], axis=3)
				# print(new_node.shape)
				if cnt == node_num:
					X_interp = new_node
				else:
					X_interp = np.concatenate((X_interp, new_node), axis=0)
					# print(X_interp.shape)
					# print(i, j)
				# print(i, j, cnt)
				cnt += 1
	# print(X_interp.shape)
	X_interp = np.reshape(X_interp, [X.shape[0], X.shape[1],X.shape[2], node_num-1, 2])
	X_interp = np.concatenate((X, X_interp), axis=-2)
	return X_interp

"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
	nb_graphs = adj.shape[0]
	mt = np.empty(adj.shape)
	for g in range(nb_graphs):
		mt[g] = np.eye(adj.shape[1])
		for _ in range(nhood):
			mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
		for i in range(sizes[g]):
			for j in range(sizes[g]):
				if mt[g][i][j] > 0.0:
					mt[g][i][j] = 1.0
	return -1e9 * (1.0 - mt)

