from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from decimal import Decimal
import pandas as pd
import numpy as np
import sys

#split data, one for training, another for testing
def split_data(dataset):
	normal = dataset[dataset['label']==0] 
	normal_train = normal[:len(normal)/2]
	normal_test = normal[len(normal)/2:]    

	mali = dataset[dataset['label']==1]
	mali_train = mali[:len(mali)/2]
	mali_test = mali[len(mali)/2:]

	train = normal_train.append(mali_train)
	test = normal_test.append(mali_test)

	x_train = train.drop(['label'], axis=1)
	y_train = train['label']
	x_test = test.drop(['label'], axis=1)
	y_test = test['label']

	return x_train, y_train, x_test, y_test

def local_optimal_feature_selection(filename):
	print("---------waiting for feature selection---------")
	data = pd.read_csv(filename)
	#delete column which are not features
	features = data.drop(['file_name'], axis=1) 

	#delete the features whose variance is equal to 0, which are irrevevant features
	selector = VarianceThreshold(threshold=0)
	selector.fit(features)  
	#retain dataframe https://stackoverflow.com/questions/39812885/retain-feature-names-after-scikit-feature-selection
	features = features[features.columns[selector.get_support(indices=True)]] 
	#print(features.shape[1])
	#features.to_csv("./csv/preprocess_data.csv", index=False)
	#sys.exit(1)

	x_train, y_train, x_test, y_test = split_data(features)
	

	tmp = 0
	tmp_ACC = 0.0
	tmp_FPR = 0.0
	tmp_F1 = 0.0
	t_train = x_train
	t_test = x_test

	#loop the number of times which is equal to the number of features
	for i in range(0, features.shape[1]-1): 
		x_train = x_train.drop(x_train.columns[tmp], axis=1)
		x_test = x_test.drop(x_test.columns[tmp], axis=1)

		model = RandomForestClassifier(n_estimators=100,n_jobs=-1, random_state=0) 
		model.fit(x_train, y_train)
		y_pre = model.predict(x_test)
		cnf_matrix = confusion_matrix(y_test, y_pre)
	
		TN = cnf_matrix[0][0]
		FP = cnf_matrix[0][1]
		FN = cnf_matrix[1][0]
		TP = cnf_matrix[1][1]
		ACC = Decimal((TP + TN)) / (TP + TN + FP + FN)
		FPR = Decimal(FP) / (TN + FP)
		Precision = Decimal(TP) / (TP+FP)
		Recall = Decimal(TP) / (TP+FN)
		F1 = 2*Precision*Recall / (Precision+Recall)
	
		if i == 0:
			tmp_ACC = ACC
			tmp_FPR = FPR
			tmp_F1 = F1
			continue
		if ACC >= tmp_ACC or FPR <= tmp_FPR: #or F1 >= F1_tmp 
			t_train = x_train
			t_test = x_test
			tmp_ACC = ACC
			tmp_FPR = FPR
			tmp_F1 = F1
			tmp_matrix = cnf_matrix
			print('Local optimal result:')
			print(tmp_matrix)
		else:
			x_train = t_train
			x_test = t_test
			tmp = tmp + 1

	# feature importance in descending order
	sorted_idx = model.feature_importances_.argsort()[::-1] 
	print('the number of selected features:%d' % x_train.shape[1])

 	#formatted output the feature_importance.txt
	tplt = "{0:^50}\t{1:^20}"
	with open('./l_feature_importances.txt', 'a') as f:
		print >>f, tplt.format("feature name", "importance")
		for idx in sorted_idx:
			print >>f, tplt.format(x_train.columns[idx], model.feature_importances_[idx])
		f.close()

	print('best ACC after feature selection: %.5f' % tmp_ACC)
	print('best FPR after feature selection: %.5f' % tmp_FPR)
	print('best F1 after feature selection: %.5f' % tmp_F1)

	data1 = pd.concat([x_train,y_train], axis=1)
	data2 = pd.concat([x_test,y_test], axis=1)

	selected_data = data1.append(data2)
	selected_data.to_csv("./csv/feature_selected_data.csv", index=False)


def evaluate_model(data, estimators, m_features, m_depth):
	x_train, y_train, x_test, y_test = split_data(data)

	model = RandomForestClassifier(n_estimators=estimators, max_features=m_features, max_depth=m_depth, min_samples_split=2, min_samples_leaf=1, n_jobs=-1,random_state=0) 
	model.fit(x_train, y_train)
	y_pre = model.predict(x_test)
	cnf_matrix = confusion_matrix(y_test, y_pre)

	TN = cnf_matrix[0][0]
	FP = cnf_matrix[0][1]
	FN = cnf_matrix[1][0]
	TP = cnf_matrix[1][1]
	ACC = Decimal((TP + TN)) / (TP + TN + FP + FN)
	FPR = Decimal(FP) / (TN + FP)

	return cnf_matrix, ACC, FPR

def manually_debugging_options(filename):
	print("---------mannually debugging options---------")
	data = pd.read_csv(filename)
	X = data.drop(['label'], axis=1)
	y = data['label']
 
	#mannually set options
	cnf_matrix, ACC, FPR =evaluate_model(data, 100, 'auto', m_depth=None)  
	print('the performance of the models:')
	print(cnf_matrix)
	print('the ACC : %.5f' % ACC)
	print('the FPR : %.5f' % FPR)

#debugging options by using greedy coordinate descent method
def automatically_debugging_options(filename):
	print("---------waiting for automatically debugging options---------")
	data = pd.read_csv(filename)
	#print(data.shape)
	#sys.exit(1)
	X = data.drop(['label'], axis=1)
	y = data['label']
 
	#the model before debugging
	cnf_matrix, ACC, FPR =evaluate_model(data, 100, 'auto', m_depth=None)
	print('the performance of the model before debugging options:')
	print(cnf_matrix)
	print('the ACC : %.5f' % ACC)
	print('the FPR : %.5f' % FPR)
	
	#select best n_estimators
	param_test = {'n_estimators':range(70,90,2)}
	grid_search = GridSearchCV(estimator = RandomForestClassifier(n_jobs=-1,random_state=0), param_grid = param_test,
                                scoring='roc_auc',verbose=1, n_jobs=-1, cv=KFold(3, random_state=0))
	grid_search.fit(X, y)
	best_estimator =grid_search.best_params_['n_estimators']
	print('the best n_estimators: %d' % best_estimator)


	#select best max_features
	param_test = {'max_features':range(6,24,2)}
	grid_search = GridSearchCV(estimator = RandomForestClassifier(n_estimators=best_estimator,n_jobs=-1,random_state=0), 
                               param_grid = param_test, scoring='roc_auc',verbose=1, n_jobs=-1, cv=KFold(3, random_state=0))
	grid_search.fit(X, y)
	best_max_features =grid_search.best_params_['max_features']
	print('the best max_features: %d' % best_max_features)

			
	#select best max_depth
	param_test = {'max_depth':range(12,24,2)}
	grid_search = GridSearchCV(estimator = RandomForestClassifier(n_estimators=best_estimator, max_features=best_max_features, n_jobs=-1,random_state=0), 
                               param_grid = param_test, scoring='roc_auc',verbose=1, n_jobs=-1, cv=KFold(3, random_state=0))
	grid_search.fit(X, y)
	best_max_depth =grid_search.best_params_['max_depth']
	print('the best max_depth: %d' % best_max_depth)
	

	#select best min_samples_split
	param_test = {'min_samples_split':range(2,11,1)}
	grid_search = GridSearchCV(estimator = RandomForestClassifier(n_estimators=best_estimator, max_features=best_max_features, max_depth=best_max_depth, n_jobs=-1,random_state=0), 
                               param_grid = param_test, scoring='roc_auc',verbose=1, n_jobs=-1, cv=KFold(3, random_state=0))
	grid_search.fit(X, y)
	best_min_samples_split =grid_search.best_params_['min_samples_split']
	print('the best min_samples_split: %d' % best_min_samples_split)
	

	#select best min_samples_leaf
	param_test = {'min_samples_leaf':range(1,10,1)}
	grid_search = GridSearchCV(estimator = RandomForestClassifier(n_estimators=best_estimator, max_features=best_max_features, max_depth=best_max_depth, min_samples_split=best_min_samples_split, n_jobs=-1,random_state=0), 
                               param_grid = param_test, scoring='roc_auc',verbose=1, n_jobs=-1, cv=KFold(3, random_state=0))
	grid_search.fit(X, y)
	best_min_samples_leaf =grid_search.best_params_['min_samples_leaf']
	print('the best min_samples_leaf: %d' % best_min_samples_leaf)


	#evaluate the model after debugging options
	cnf_matrix, ACC, FPR =evaluate_model(data, best_estimator, best_max_features, best_max_depth)
	print('the performance of the model after debugging options:')
	print(cnf_matrix)
	print('the ACC : %.5f' % ACC)
	print('the FPR : %.5f' % FPR)
	

if __name__ == '__main__':
	filename1 = 'csv/original_data.csv'
	filename2 = 'csv/feature_selected_data.csv'
	
	local_optimal_feature_selection(filename1)

	automatically_debugging_options(filename2)

	#manually_debugging_options(filename2)