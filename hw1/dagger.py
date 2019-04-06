import gym
import tensorflow as tf
import load_policy
import tf_util
import pickle
import numpy as np
import myUtil
import load_policy
import matplotlib.pyplot as plt


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

def saveNewData(obs,actions):
	fileName = "/home/yousof/courses/cs294_DeepRL/hw1/expert_data/"+model+"_newDataForDagger.pkl"
	toSave = dict()
	toSave["newObs"] = obs
	toSave["newActs"] = actions
	save = pickle.dump(toSave, open(fileName, "wb" ))

if __name__ == "__main__":
	model = "Humanoid-v2"
	logfile = "logfile"
	model_path = "models/" + model
	expert_policy_file = "/home/yousof/courses/cs294_DeepRL/hw1/experts/" + model + ".pkl"
	render = False
	num_rollouts = 100
	max_steps = 1000
	reward_th = 8

	x_mean, x_std, y_mean, y_std = myUtil.load_normalization(model)
	print("shape of x_std", x_std[(x_std!=0)].shape)

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
		extra_data_for_dagger_obs = []
		extra_data_for_dagger_acts = []

		plt.ion()
		plt.show()


		for i in range(num_rollouts):
			for j in range (311,314):
				plt.subplot(j)
				plt.cla()
			obs = env.reset()
			observations.append(obs)
			done = False
			totalr = 0
			steps = 0
			dones = []
			steps_after_failure = 0 
			rewards_for_expert_policy = []
			rewards_for_my_policy = []

			model_in = prepareInput(obs, obsPre, obs_len, x_mean, x_std)
			obsPre = obs
			preAction = run_policy(model_in)
			action = prepareOutput(preAction, y_mean, y_std)
			obs, r, done, _ = env.step(action)
			while True: # not done
				predone = done
				model_in = prepareInput(obs, obsPre, obs_len, x_mean, x_std)
				obsPre = obs
				if not done and steps_after_failure%150 == 0:#
					preAction = run_policy(model_in)
					action = prepareOutput(preAction, y_mean, y_std)
					obs, r, done, _ = env.step(action)
					rewards_for_expert_policy.append(0)
					rewards_for_my_policy.append(r)
				else:
					steps_after_failure += 1
					action = expert_policy_fn(obs[None,:])
					obs, r, done, _ = env.step(action)
#					print ("running Expert!", done)
					if (r>reward_th):
						extra_data_for_dagger_obs.append(prepareInput(obs, obsPre, obs_len, x_mean, x_std))
						extra_data_for_dagger_acts.append(myUtil.normalize(action,y_mean, y_std))
					rewards_for_expert_policy.append(r)
					rewards_for_my_policy.append(0)
				dones.append(done)
#				if (predone == True and done == False): 
#					print("yes, after finishing, it can come back")
				totalr += r
				steps += 1
				if render:
					env.render()
				if steps % 100 == 0: 
					print("%i/%i"%(steps, max_steps),end="\r")
				if steps >= max_steps:
					break
				i += 1
#				if i%50 ==0:
			plt.subplot(311)
			plt.plot(np.array(dones)*10,'b')
			plt.subplot(312)
			plt.plot(rewards_for_expert_policy,'b')
			plt.subplot(313)
			plt.plot(rewards_for_my_policy,'b')
			plt.draw()
			plt.pause(0.001)
			returns.append(totalr)
#			input("Press [enter] to continue.")
		extra_data_for_dagger_obs = np.array(extra_data_for_dagger_obs)
		extra_data_for_dagger_acts = np.array(extra_data_for_dagger_acts)
		saveNewData(extra_data_for_dagger_obs,extra_data_for_dagger_acts)
		print("shape of new observations", extra_data_for_dagger_obs.shape)
		print("shape of new actions", extra_data_for_dagger_acts.shape)
		print('returns', returns)
		print('mean return', np.mean(returns))
		print('std of return', np.std(returns))