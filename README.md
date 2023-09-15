# ModelDeployer
Scripts to deploy PyTorch Async Inference SageMaker models for [FINNEAS.AI](https://www.finneas.ai/)

## Setup
Install dependencies for deploying to SageMaker.
```console
pip install -r requirements.txt
```

## Deploy a model
To deploy a model to FINNEAS.AI requires three things:
- Deployment config file
- Source code for the model
- Model file
The deployment config file contains paths to the source code and the model file.
The model can be deployed by running a command like so:
```console
python main.py deploy <deployment-config-file.json>
```
This will deploy the model to a SageMaker Endpoint with autoscaling enabled.

### Deployment config file
The deployment config file needs to be in a JSON file.
An example config file looks like this:
```
{
  "code_path": "PTNet3D/",
  "pytorch_model_file": "model/T2_to_T1.pth",
  "zipped_model_path": "model.tar.gz",
  "version": "0.1.2",
  "endpoint_name": "PTNet3D-T2-to-T1-v0-1-2",
  "entry_point": "PTNet3D/inference_code.py",
  "bucket_prefix": "PTNet3D-async-inference",
  "instance_type": "ml.g4dn.xlarge",
  "model": "Convert T2 to T1",
  "source_code_url": "https://github.com/XuzheZ/PTNet3D",
  "task": "Modality Transfer",
  "FilenameSchema": "T1toT2"
}
```
JSON Fields:
- __code_path__ - path to source code for your model to be packaged for SageMaker
- __pytorch_model_file__ - path to pytorch model to be packaged for SageMaker
- __zipped_model_path__ - path to where we save the SageMaker packaged model (this can always be model.tar.gz)
- __version__ - version of your model
- __endpoint_name__ - unique name for this model/version number
- __entry_point__ - relative path to the sagemaker specific
- __bucket_prefix__ - prefix used to store job files used by sagemaker 
- __instance_type__ - AWS instance type used to perform inference
- __model__ - value in Model dropdown on Launch Task page (this value stored as a "Model" tag on the endpoint)
- __source_code_url__ - source code URL displayed in Model Details on Launch Task page (this value stored as a "SourceCodeURL" tag on the endpoint)
- __task__ - value in Task dropdown on Launch Task page (this value stored as a "Task" tag on the endpoint)
- __filename_schema__ - optional field to specify input and output filename transformations (only 

## Configure Autoscaling
To enable autoscaling run:
```console
python main.py autoscale <endpoint-name>
```

For troubleshooting or speeding up preductions for a known heavy usage time autoscaling can be disabled.

To disable autoscaling run:
```console
python main.py autoscale --disable <endpoint-name>
```
