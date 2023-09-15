import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker import get_execution_role
import tarfile
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
import boto3
import logging


def deploy_model(modelconfig):
    code_path = modelconfig["code_path"]
    pytorch_model_file = modelconfig["pytorch_model_file"]
    zipped_model_path = modelconfig["zipped_model_path"]
    endpoint_name = modelconfig["endpoint_name"]
    entry_point = modelconfig["entry_point"]
    bucket_prefix = modelconfig["bucket_prefix"]
    instance_type = modelconfig["instance_type"]
    sm_role = sagemaker.get_execution_role()
    tags = [
        {
            "Key":"Role",
            "Value": sm_role
        },
        {
            "Key":"Model",
            "Value": modelconfig["model"]
        },
        {
            "Key":"SourceCodeURL",
            "Value": modelconfig["source_code_url"]
        },
        {
            "Key":"Task",
            "Value": modelconfig["task"]
        },
        {
            "Key":"Version",
            "Value": modelconfig["version"]
        },
        {
            "Key":"FilenameSchema",
            "Value": modelconfig["filename_schema"]
        }
    ]

    print(f"Creating {zipped_model_path} from {pytorch_model_file} and {code_path}.")
    with tarfile.open(zipped_model_path, "w:gz") as tar:
        tar.add(pytorch_model_file, arcname="current.pth")
        tar.add(code_path, arcname="code")

    print(f"Creating model {endpoint_name}.")
    model = PyTorchModel(
        name=endpoint_name,
        entry_point=entry_point,
        model_data=zipped_model_path,
        role=sm_role,
        framework_version="1.11.0",
        py_version="py38",
        container_log_level=logging.DEBUG,
        env={'SAGEMAKER_MODEL_SERVER_TIMEOUT': '3600'}
    )
    boto_session = boto3.session.Session()
    sm_session = sagemaker.session.Session()
    print(f"Using Role: {sm_role}")
    s3_bucket = sm_session.default_bucket()
    async_config = AsyncInferenceConfig(
        output_path=f"s3://{s3_bucket}/{bucket_prefix}/output",
        max_concurrent_invocations_per_instance=4,
        notification_config={
            "SuccessTopic": "arn:aws:sns:us-east-1:387950165774:AsyncInferenceSuccessTopic",
            "ErrorTopic": "arn:aws:sns:us-east-1:387950165774:AsyncInferenceErrorTopic",
        }
    )
    print(f"Deploying model {endpoint_name}")
    print("    task:", modelconfig["task"])
    print("    model:", modelconfig["model"])
    print("    version:", modelconfig["version"])
    print("    instance_type:", modelconfig["instance_type"])

    predictor = model.deploy(
        async_inference_config=async_config,
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        tags=tags
    )
    print("Done")


def enable_autoscaling(endpoint_name):
    print(f"Enabling autoscaling for {endpoint_name}")
    client = boto3.client(
        "application-autoscaling"
    )  # Common class representing Application Auto Scaling for SageMaker amongst other services

    resource_id = (
        "endpoint/" + endpoint_name + "/variant/" + "AllTraffic"
    )  # This is the format in which application autoscaling references the endpoint

    # Configure Autoscaling on asynchronous endpoint down to zero instances
    response = client.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=5,
    )
    response = client.put_scaling_policy(
        PolicyName="Invocations-ScalingPolicy",
        ServiceNamespace="sagemaker",  # The namespace of the AWS service that provides the resource.
        ResourceId=resource_id,  # Endpoint name
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",  # SageMaker supports only Instance Count
        PolicyType="TargetTrackingScaling",  # 'StepScaling'|'TargetTrackingScaling'
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 0.9,  # The target value for the metric.
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSize",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Statistic": "Average",
            },
            "ScaleInCooldown": 600,  # The cooldown period helps you prevent your Auto Scaling group from launching or terminating
            # additional instances before the effects of previous activities are visible.
            # You can configure the length of time based on your instance startup time or other application needs.
            # ScaleInCooldown - The amount of time, in seconds, after a scale in activity completes before another scale in activity can start.
            "ScaleOutCooldown": 300  # ScaleOutCooldown - The amount of time, in seconds, after a scale out activity completes before another scale out activity can start.
            # 'DisableScaleIn': True|False - ndicates whether scale in by the target tracking policy is disabled.
            # If the value is true , scale in is disabled and the target tracking policy won't remove capacity from the scalable resource.
        },
    )
    print("Done")


def disable_autoscaling(endpoint_name):
    print(f"Disabling autoscaling for {endpoint_name}")
    client = boto3.client(
        "application-autoscaling"
    )  # Common class representing Application Auto Scaling for SageMaker amongst other services
    resource_id = (
        "endpoint/" + endpoint_name + "/variant/" + "AllTraffic"
    )  # This is the format in which application autoscaling references the endpoint
    response = client.deregister_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    )
    print("Done")
