import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import preprocessing
from surprise import accuracy
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import NMF
from surprise import KNNBasic
from surprise.model_selection import cross_validate

import git
import sys
repo = git.Repo("./", search_parent_directories=True)
homedir = repo.working_dir
sys.path.insert(1, f"{homedir}" + '/models/data_processing')
import loader


def normalize(lst):
	lst = np.array(lst)
	mean = np.mean(lst)
	return 2*lst/mean

def trainPCA(data, colorlabels, sizelabels, plot=True, savefig=None):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(data)
	first = principalComponents[:,0]
	second = principalComponents[:,1]

	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('Principal Component 1', fontsize = 15)
		ax.set_ylabel('Principal Component 2', fontsize = 15)
		ax.set_title('2 component PCA', fontsize = 20)
		scatter = ax.scatter(first, second, c = colorlabels, s = sizelabels)
		ax.grid()
		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_label("state")
		if savefig:
			plt.savefig("figures/"+"pca_"+savefig)
		plt.show()

def trainSVD(data, counties, colorlabels, sizelabels, plot=True, savefig=None): #colorlabels, sizelabels, plot=True, savefig=True
	min_max_scaler = preprocessing.MinMaxScaler()
	data = min_max_scaler.fit_transform(data)

	reformatted = []
	for i, parameters in enumerate(data):
		county = int(counties[i])
		for j, param in enumerate(parameters):
			reformatted.append([county, j, param])
	df = pd.DataFrame(reformatted) #column0 is fips, column1 is param_id (0-16), column2 is param_value

	reader = Reader(rating_scale=(0, 1))
	data = Dataset.load_from_df(df[[0,1,2]], reader)
	trainset = data.build_full_trainset() # Not doing cross validation, but maybe try that too
	# algo = NMF(n_factors=2, n_epochs=100)
	# algo = KNNBasic()
	algo = SVD(n_factors=4, n_epochs=1000, biased=True)
	algo.fit(trainset)
	U = algo.pu 
	if plot:
		fig = plt.figure(figsize = (8,8))
		ax = fig.add_subplot(1,1,1) 
		ax.set_xlabel('First', fontsize = 15)
		ax.set_ylabel('Second', fontsize = 15)
		ax.set_title('Reduced SVD', fontsize = 20)
		scatter = ax.scatter(U[:,0], U[:,1], c = colorlabels, s = sizelabels)
		ax.grid()
		cbar = fig.colorbar(scatter, ax=ax)
		cbar.set_label("state")
		if savefig:
			plt.savefig("figures/"+"svd"+savefig)
		plt.show()

	# U = U.transpose()
	# A = np.linalg.svd(U)[0]
	# U_proj = np.dot(A[:, :2].transpose(), U)
	# # Rescale dimensions
	# U_proj /= U_proj.std(axis=1).reshape(2, 1)

	# if plot:
	# 	fig = plt.figure(figsize = (8,8))
	# 	ax = fig.add_subplot(1,1,1) 
	# 	ax.set_xlabel('First', fontsize = 15)
	# 	ax.set_ylabel('Second', fontsize = 15)
	# 	ax.set_title('Reduced SVD', fontsize = 20)
	# 	scatter = ax.scatter(U[0], U[1], c = colorlabels, s = sizelabels)
	# 	ax.grid()
	# 	cbar = fig.colorbar(scatter, ax=ax)
	# 	cbar.set_label("state")
	# 	if savefig:
	# 		plt.savefig("figures/"+"svd"+savefig)
	# 	plt.show()

def trainTSNE(data, counties, colorlabels, sizelabels, plot=True, savefig=None): #colorlabels, sizelabels, plot=True, savefig=True
	# min_max_scaler = preprocessing.MinMaxScaler()
	# data = min_max_scaler.fit_transform(data)

	datafile = '' #get this either from command line argument or encapslate into function that can be imported in other files
	X = data
	y = counties
	print (X.shape, y.shape)
	feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
	df = pd.DataFrame(X,columns=feat_cols)
	df['label'] = y
	df['label'] = df['label'].apply(lambda i: str(i))
	X, y = None, None
	print ('Size of the dataframe: {}'.format(df.shape))
	rndperm = np.random.permutation(df.shape[0])
	n_sne = len(data)
	tsne = TSNE(n_components=3, verbose=0, perplexity=30, n_iter=1000) #n_components=2, verbose=1, perplexity=40, n_iter=300
	tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
	df_tsne = df.loc[rndperm[:n_sne],:].copy()
	df_tsne['x-tsne'] = tsne_results[:,0]
	df_tsne['y-tsne'] = tsne_results[:,1]
	# Create the figure
	if plot:
		fig = plt.figure( figsize=(8,8) )
		ax = fig.add_subplot(1, 1, 1, title='TSNE' )
		# Create the scatter
		ax.scatter(
		    x=df_tsne['x-tsne'], 
		    y=df_tsne['y-tsne'], 
		    s=sizelabels,
		    c=colorlabels, 
		    cmap=plt.cm.get_cmap('Paired'), 
		    alpha=0.7)
		if savefig:
				plt.savefig("figures/"+"tsne"+savefig)
		plt.show()

def visualize(input_path):
	input_file = Path(input_path).stem
	fopen = open(input_path,'r')
	data = fopen.read()
	json_decoded = data.replace("'", "\"")
	params_dict = json.loads(json_decoded)

	populations = loader.load_data("/data/us/demographics/county_populations.csv")
	densities = loader.load_data("/data/us/demographics/county_land_areas.csv", encoding='latin-1')

	X = np.array(list(params_dict.values()))
	X = X[:,0,0:17]
	counties = np.array(list(params_dict.keys()))
	for index, county in enumerate(counties):
		if len(county) == 4:
			counties[index] = '0'+county
	states_labels = [int(fips[0:2]) for fips in counties]
	pop_labels = []
	density_labels = []
	for county in counties:
		population = (loader.query(populations, "FIPS", int(county))["total_pop"].values)[0]
		density  = (loader.query(densities, "County FIPS", int(county))['2010 Density per square mile of land area - Population'].values)[0]
		pop_labels.append(population)
		density_labels.append(density)

	pop_labels = normalize(pop_labels)
	density_labels = normalize(density_labels)
	trainPCA(X, states_labels, density_labels, savefig=input_file)
	trainSVD(X, counties, states_labels, density_labels, savefig=input_file)
	trainTSNE(X, counties, states_labels, density_labels, savefig=input_file)

if __name__ == '__main__':
	input_path = '../old_parameters2_1_0.csv'
	visualize(input_path)


