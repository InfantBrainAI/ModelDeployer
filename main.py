#!/usr/bin/env python
import json
import click
from util import deploy_model, disable_autoscaling, enable_autoscaling


@click.group()
def cli():
    pass


@click.command()
@click.argument('model_config')
def deploy(model_config):
    "Deploy a model (or a new version of an existing model) to SageMaker"
    with open(model_config) as infile:
        deploy_model(json.load(infile))


@click.command()
@click.argument('endpoint_name')
@click.option('--disable', 'disable', is_flag=True, help="Disable autoscaling")
def autoscale(endpoint_name, disable):
    "Enable or disable autoscaling for a SageMaker endpoint"
    if disable:
        disable_autoscaling(endpoint_name)
    else:
        enable_autoscaling(endpoint_name)

cli.add_command(deploy)
cli.add_command(autoscale)


def main():
    try:
        cli()
    except Exception as e:
        click.echo(str(e))


if __name__ == '__main__':
    main()
