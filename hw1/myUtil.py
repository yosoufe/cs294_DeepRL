import numpy as np
import pickle

def load_normalization(model):
	"""
	model a string containing the model name.
	it returns  (x_mean, x_std, y_mean, y_std)
	"""
	loaded = pickle.load( open( "/home/yousof/courses/cs294_DeepRL/hw1/expert_data/"+
		model+"_normalization.pkl", "rb" ) )
	return (loaded["x_mean"],loaded["x_std"],loaded["y_mean"],loaded["y_std"])

def normalize(x,mean=None,std=None):
	"""
	If the Mean is not give, it calculates the mean and std
	otherwise it normalizes using the mean and std 
	"""
	if (mean is None):
		mean = np.mean(x,axis=0)
		std = np.std(x,axis=0)
		normalized = (x-mean)/std
		return normalized, mean, std
	else:
		return (x-mean)/std

def unnormalize(x, mean, std):
	unnorm = x * std + mean
	return unnorm

def removeUselessData(x, std):
	"""
	It removes the dimensions that the std is zero
	"""
	assert (x.shape[-1] == std.shape[-1]), "The shapes of std and x does not match"
	numberOfSampels = x.shape[0]
	mask = (std!=0)
	t = np.tile(mask,(numberOfSampels,1))
	return x[t].reshape([numberOfSampels,-1])

def isAllDataUsefull(x):
	return not (np.isnan(x).any() and np.isinf(x).any())

def addDiff2Data(inp, batch_size):
	"""
	Just assumes the every batch_size there is one rollout
	For example if we assume that the data is only one rollout,
	the output's zeroth dimension would be 1 less.
	in general the out.shape[0] would be inp.shape[0]-batch_size
	Therefore it removes the first observation from the data.
	"""
	num_ins = inp.shape[1]
	out = np.array([], dtype=np.float16).reshape(0,num_ins * 2 )
	for idx, i in enumerate(range(0,num_examples,bat)):
		start = i
		end = i+bat
		xpreK = x_i[start:end-1,:]
		xK = x_i[start+1:end,:]
		out = np.concatenate( (out,np.concatenate((xK, xK-xpreK ) , axis = 1) ), axis = 0)
	return out

if __name__ == "__main__":
	model = "Humanoid-v2"
	x_mean, x_std, y_mean, y_std = load_normalization(model)
	print (y_std.shape)

print("Hello from here")