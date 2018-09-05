import gym
import tensorflow as tf
import load_policy
import tf_util
import pickle
import numpy as np
import myUtil
import load_policy


def prepareInput(obs, preObs, size, x_mean, x_std):
	x_mask = (x_std!=0)
	obs =obs.reshape([1,size])
	preObs = preObs.reshape([1,size])
	model_in = np.concatenate( (obs,obs-preObs), axis = 1 )
	model_in = myUtil.normalize(model_in, mean = x_mean, std = x_std)
	model_in = model_in[x_mask.reshape([1,size*2])]
	model_in = np.expand_dims(model_in, axis=0)
	return model_in

def prepareOutput(modelOut, y_mean, y_std ):
    return myUtil.unnormalize(modelOut,y_mean,y_std)


if __name__ == "__main__":
	model = "Humanoid-v2"
	logfile = "logfile"
	model_path = "models/" + model
	expert_policy_file = "/home/yousof/courses/cs294_DeepRL/hw1/experts/" + model + ".pkl"
	render = True
	num_rollouts = 2
	max_steps = 1000

	x_mean, x_std, y_mean, y_std = myUtil.load_normalization(model)

	print ("load the expert policy funciton")
	expert_policy_fn = load_policy.load_policy(expert_policy_file)
	print("expert policy funciton loaded and built")

	with tf.Session() as sess:
		# load the model
		saver = tf.train.import_meta_graph(model_path + '.ckpt.meta')
		saver.restore(sess,model_path+".ckpt")
		graph = tf.get_default_graph()
		policy_out = graph.get_tensor_by_name('pred/Tanh:0')#I_want_to_train_only_these/fc8/BiasAdd:0
		x_place = graph.get_tensor_by_name('input:0')

		def run_policy(policy_in):
		    return sess.run(policy_out, feed_dict = {x_place: policy_in})

		env = gym.make(model)

		obs_len = env.observation_space.shape[0]
		obsPre = np.zeros((1,obs_len),dtype=np.float16)

		
		returns = []
		observations = []
		actions = []

		for i in range(num_rollouts):
			obs = env.reset()
			observations.append(obs)
			done = False
			totalr = 0
			steps = 0

			model_in = prepareInput(obs, obsPre, obs_len, x_mean, x_std)
			obsPre = obs
			preAction = run_policy(model_in)
			action = prepareOutput(preAction, y_mean, y_std)
			obs, r, done, _ = env.step(action)
			while True: # not done
				predone = done
				model_in = prepareInput(obs, obsPre, obs_len, x_mean, x_std)
				obsPre = obs
				if not done:
					preAction = run_policy(model_in)
					action = prepareOutput(preAction, y_mean, y_std)
				else:
					action = expert_policy_fn(obs[None,:])
					print ("running Expert!")
				obs, r, done, _ = env.step(action)
				if (predone == True and done == False): 
					print("yes, after finishing, it can come back")
				totalr += r
				steps += 1
				if render:
					env.render()
				if steps % 100 == 0: 
					print("%i/%i"%(steps, max_steps),end="\r")
				if steps >= max_steps:
					break
			returns.append(totalr)
		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))