import joblib
from azureml.core.model import Model


def init(model_path):
    global model
    print("Model Path is  ", model_path)
    model = joblib.load(model_path)


def run(data, model_path):
    init(model_path)
    try:
        result = model.predict(data['data'])
        return {'data' : result.tolist() , 'message' : "Successfully classified Iris"}
    except Exception as e:
        error = str(e)
        return {'data' : error , 'message' : 'Failed to classify iris'}



