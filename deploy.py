from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice, Webservice


def deploy(ws_name,model_name,path_to_model, 
           environment_name,register_environment,pip_packages,conda_packages,
           cpu_cores , memory_gb, path_to_entry_script,service_name):

    '''
        Get Workspace
    '''
    ws = Workspace.from_config()
    print("Got Workspace {}".format(ws_name))


    '''
        Register Model
    '''
    model = Model.register(workspace = ws,
                        model_path =path_to_model,
                        model_name = model_name,
                        )
    print("Registered Model {}".format(model_name))

    '''
        Register Environment
    '''
    if register_environment:
        env = Environment(environment_name)
        cd = CondaDependencies.create(pip_packages=pip_packages, conda_packages = conda_packages)
        env.python.conda_dependencies = cd
        # Register environment to re-use later
        env.register(workspace = ws)
        print("Registered Environment")
    myenv = Environment.get(workspace=ws, name=environment_name)
    
    myenv.save_to_directory('./environ', overwrite=True)

    '''
        Config Objects
    '''
    aciconfig = AciWebservice.deploy_configuration(
            cpu_cores=1,
            memory_gb=1,
            tags={"data":"iris classifier"},
            description='iRIS cLASSIFICATION knn MODEL',
        )
    inference_config = InferenceConfig(entry_script=path_to_entry_script, environment=myenv) 

    '''
        Deploying
    '''

    print("Deploying....... This may take a few mins, check the status in MLS after the function finishes executing")
    service = Model.deploy(workspace=ws, 
                        name=ws_name, 
                        models=[model], 
                        inference_config=inference_config, 
                        deployment_config=aciconfig, overwrite = True)

    service.wait_for_deployment(show_output=True)
    url = service.scoring_uri    
    print(url)

    service = Webservice(ws,ws_name)
    print(service.get_logs()) 

    return url
  