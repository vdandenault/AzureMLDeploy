import pickle

import joblib
import pandas as pd
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from deploy import deploy
from score import run

data = load_iris()
df = pd.DataFrame(data.data , columns = data.feature_names)
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
X_train , X_test, y_train, y_test = train_test_split(df , data.target , test_size = 0.2)

train_scores , test_scores = [] , []
for k in range(1,10):
    model = KNeighborsClassifier(n_neighbors = k)
    model.fit(X_train , y_train)
    train_score = model.score(X_train, y_train)
    train_scores.append(train_score)
    test_score = model.score(X_test , y_test)
    test_scores.append(test_score)

# To dump
fileName = 'model_knn.pkl'
pickle.dump( model, open(fileName, "wb" ) )
# To load
loaded_model = joblib.load(fileName)
loaded_model.predict(scaler.fit_transform( [[2,0,2,10]]))
try: 
    ws = Workspace.create(
               name='myworkspace-test',            
               subscription_id='d39d58d1-51e7-451f-a27c-0696ddf4bfaf',           
               resource_group='myresourcegroup',                 
               create_resource_group=True,                 
               location='eastus2'                
            )
    ws.write_config()
except Exception as e: 
    ws = Workspace.from_config()
    print("Got Workspace {}".format(ws.name))
    aciconfig = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            tags={"data":"iris classifier"},
            description='iRIS cLASSIFICATION knn MODEL',
            )
    deploy(ws.name, 'KNN', fileName, environment_name="mytestenvironment",register_environment=False,pip_packages=[],conda_packages=[],
            cpu_cores=1 , memory_gb=1, path_to_entry_script=fileName,service_name=aciconfig.description)


run(data, fileName)
