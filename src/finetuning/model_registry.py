"""
Model Registry for MLflow and W&B Integration.
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import mlflow
import wandb
from loguru import logger
from mlflow.tracking import MlflowClient

# Optional W&B import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("Weights & Biases not available. Install with: pip install wandb")


@dataclass
class ModelMetadata:
    """Model metadata for registry."""

    model_name: str
    model_path: str
    base_model: str
    training_config: Dict[str, Any]
    evaluation_metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    hardware_info: Dict[str, Any]
    training_time: float
    model_size: str
    version: str = "1.0.0"


class ModelRegistry:
    """Model registry for MLflow and W&B."""

    def __init__(self, experiment_name: str = "intelligent-research-assistant"):
        """
        Initialize model registry.

        Args:
            experiment_name: Name of the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.mlflow_client = MlflowClient()

        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)

        logger.info(f"Model registry initialized with experiment: {experiment_name}")

    def register_model_mlflow(
        self,
        model_metadata: ModelMetadata,
        model_path: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Register model in MLflow.

        Args:
            model_metadata: Model metadata
            model_path: Path to the model files
            tags: Additional tags for the model

        Returns:
            Model URI
        """
        logger.info(f"Registering model in MLflow: {model_metadata.model_name}")

        # Start MLflow run
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(model_metadata.training_config)

            # Log metrics
            mlflow.log_metrics(model_metadata.evaluation_metrics)

            # Log model files
            mlflow.log_artifacts(model_path, "model")

            # Log additional metadata
            mlflow.log_dict(
                {
                    "dataset_info": model_metadata.dataset_info,
                    "hardware_info": model_metadata.hardware_info,
                    "training_time": model_metadata.training_time,
                },
                "metadata.json",
            )

            # Set tags
            run_tags = {
                "model_name": model_metadata.model_name,
                "base_model": model_metadata.base_model,
                "model_size": model_metadata.model_size,
                "version": model_metadata.version,
                "framework": "transformers",
                "fine_tuning_method": "lora",
            }

            if tags:
                run_tags.update(tags)

            mlflow.set_tags(run_tags)

            # Register model
            model_uri = f"runs:/{run.info.run_id}/model"
            registered_model = mlflow.register_model(
                model_uri=model_uri, name=model_metadata.model_name
            )

            logger.info(f"Model registered successfully: {registered_model.name}")
            return registered_model.name

    def register_model_wandb(
        self,
        model_metadata: ModelMetadata,
        model_path: str,
        project_name: str = "intelligent-research-assistant",
    ) -> str:
        """
        Register model in W&B.

        Args:
            model_metadata: Model metadata
            model_path: Path to the model files
            project_name: W&B project name

        Returns:
            Model artifact ID
        """
        logger.info(f"Registering model in W&B: {model_metadata.model_name}")

        # Initialize W&B
        wandb.init(
            project=project_name,
            name=f"finetune-{model_metadata.model_name}",
            config=model_metadata.training_config,
            tags=[model_metadata.model_size, "lora", "finetuning"],
        )

        # Log metrics
        wandb.log(model_metadata.evaluation_metrics)

        # Log model files
        model_artifact = wandb.Artifact(
            name=model_metadata.model_name,
            type="model",
            description=f"Fine-tuned {model_metadata.base_model} with LoRA",
        )

        model_artifact.add_dir(model_path)
        wandb.log_artifact(model_artifact)

        # Log additional metadata
        wandb.log(
            {
                "dataset_info": model_metadata.dataset_info,
                "hardware_info": model_metadata.hardware_info,
                "training_time": model_metadata.training_time,
                "model_size": model_metadata.model_size,
                "version": model_metadata.version,
            }
        )

        # Finish W&B run
        wandb.finish()

        logger.info(
            f"Model registered successfully in W&B: {model_metadata.model_name}"
        )
        return model_metadata.model_name

    def list_registered_models(
        self, registry_type: str = "mlflow"
    ) -> List[Dict[str, Any]]:
        """
        List registered models.

        Args:
            registry_type: Type of registry (mlflow or wandb)

        Returns:
            List of registered models
        """
        if registry_type == "mlflow":
            return self._list_mlflow_models()
        elif registry_type == "wandb":
            return self._list_wandb_models()
        else:
            raise ValueError(f"Unsupported registry type: {registry_type}")

    def _list_mlflow_models(self) -> List[Dict[str, Any]]:
        """List models registered in MLflow."""
        models = []

        try:
            registered_models = self.mlflow_client.search_registered_models()

            for model in registered_models:
                model_info = {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "tags": dict(model.tags) if model.tags else {},
                }

                # Get latest version
                latest_version = self.mlflow_client.get_latest_versions(model.name)
                if latest_version:
                    model_info["latest_version"] = latest_version[0].version

                models.append(model_info)

        except Exception as e:
            logger.error(f"Error listing MLflow models: {e}")

        return models

    def _list_wandb_models(self) -> List[Dict[str, Any]]:
        """List models registered in W&B."""
        models = []

        try:
            # This would require W&B API access
            # For now, return a placeholder
            logger.warning("W&B model listing requires API access")
            models = []

        except Exception as e:
            logger.error(f"Error listing W&B models: {e}")

        return models

    def get_model_version(
        self,
        model_name: str,
        version: Optional[str] = None,
        registry_type: str = "mlflow",
    ) -> Dict[str, Any]:
        """
        Get specific model version.

        Args:
            model_name: Name of the model
            version: Version number (latest if None)
            registry_type: Type of registry

        Returns:
            Model version information
        """
        if registry_type == "mlflow":
            return self._get_mlflow_model_version(model_name, version)
        elif registry_type == "wandb":
            return self._get_wandb_model_version(model_name, version)
        else:
            raise ValueError(f"Unsupported registry type: {registry_type}")

    def _get_mlflow_model_version(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get MLflow model version."""
        try:
            if version is None:
                # Get latest version
                versions = self.mlflow_client.get_latest_versions(model_name)
                if not versions:
                    raise ValueError(f"No versions found for model: {model_name}")
                model_version = versions[0]
            else:
                model_version = self.mlflow_client.get_model_version(
                    model_name, version
                )

            version_info = {
                "name": model_version.name,
                "version": model_version.version,
                "creation_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "current_stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "source": model_version.source,
            }

            # Get run information
            run = self.mlflow_client.get_run(model_version.run_id)
            version_info["metrics"] = run.data.metrics
            version_info["params"] = run.data.params
            version_info["tags"] = run.data.tags

            return version_info

        except Exception as e:
            logger.error(f"Error getting MLflow model version: {e}")
            return {}

    def _get_wandb_model_version(
        self, model_name: str, version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get W&B model version."""
        try:
            # This would require W&B API access
            logger.warning("W&B model version retrieval requires API access")
            return {}

        except Exception as e:
            logger.error(f"Error getting W&B model version: {e}")
            return {}

    def transition_model_stage(
        self, model_name: str, version: str, stage: str, registry_type: str = "mlflow"
    ) -> bool:
        """
        Transition model to a new stage.

        Args:
            model_name: Name of the model
            version: Model version
            stage: Target stage (staging, production, archived)
            registry_type: Type of registry

        Returns:
            Success status
        """
        if registry_type == "mlflow":
            return self._transition_mlflow_model_stage(model_name, version, stage)
        elif registry_type == "wandb":
            return self._transition_wandb_model_stage(model_name, version, stage)
        else:
            raise ValueError(f"Unsupported registry type: {registry_type}")

    def _transition_mlflow_model_stage(
        self, model_name: str, version: str, stage: str
    ) -> bool:
        """Transition MLflow model stage."""
        try:
            self.mlflow_client.transition_model_version_stage(
                name=model_name, version=version, stage=stage
            )
            logger.info(f"Model {model_name} version {version} transitioned to {stage}")
            return True

        except Exception as e:
            logger.error(f"Error transitioning MLflow model stage: {e}")
            return False

    def _transition_wandb_model_stage(
        self, model_name: str, version: str, stage: str
    ) -> bool:
        """Transition W&B model stage."""
        try:
            # This would require W&B API access
            logger.warning("W&B model stage transition requires API access")
            return False

        except Exception as e:
            logger.error(f"Error transitioning W&B model stage: {e}")
            return False

    def delete_model(self, model_name: str, registry_type: str = "mlflow") -> bool:
        """
        Delete a registered model.

        Args:
            model_name: Name of the model to delete
            registry_type: Type of registry

        Returns:
            Success status
        """
        if registry_type == "mlflow":
            return self._delete_mlflow_model(model_name)
        elif registry_type == "wandb":
            return self._delete_wandb_model(model_name)
        else:
            raise ValueError(f"Unsupported registry type: {registry_type}")

    def _delete_mlflow_model(self, model_name: str) -> bool:
        """Delete MLflow model."""
        try:
            self.mlflow_client.delete_registered_model(model_name)
            logger.info(f"MLflow model {model_name} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Error deleting MLflow model: {e}")
            return False

    def _delete_wandb_model(self, model_name: str) -> bool:
        """Delete W&B model."""
        try:
            # This would require W&B API access
            logger.warning("W&B model deletion requires API access")
            return False

        except Exception as e:
            logger.error(f"Error deleting W&B model: {e}")
            return False

    def create_model_card(self, model_metadata: ModelMetadata, output_path: str) -> str:
        """
        Create a model card for documentation.

        Args:
            model_metadata: Model metadata
            output_path: Path to save the model card

        Returns:
            Path to the created model card
        """
        model_card = f"""# Model Card: {model_metadata.model_name}

## Model Information
- **Model Name**: {model_metadata.model_name}
- **Base Model**: {model_metadata.base_model}
- **Version**: {model_metadata.version}
- **Model Size**: {model_metadata.model_size}
- **Framework**: Transformers
- **Fine-tuning Method**: LoRA

## Training Information
- **Training Time**: {model_metadata.training_time:.2f} seconds
- **Hardware**: {model_metadata.hardware_info.get('device', 'Unknown')}

### Training Configuration
```json
{json.dumps(model_metadata.training_config, indent=2)}
```

## Evaluation Results
```json
{json.dumps(model_metadata.evaluation_metrics, indent=2)}
```

## Dataset Information
```json
{json.dumps(model_metadata.dataset_info, indent=2)}
```

## Usage
This model is fine-tuned for instruction-following tasks in the research domain.

## Limitations
- Model performance may vary based on input domain
- Requires appropriate prompt formatting
- Limited to the training data distribution

## License
[Specify your license here]

## Citation
If you use this model, please cite:
```
@misc{{{model_metadata.model_name}}},
  title={{Fine-tuned Language Model for Research Assistant}},
  author={{Your Name}},
  year={{2025}},
  url={{[Your URL]}}
```
"""

        # Save model card
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        else:
            # If no directory specified, use current directory
            output_path = os.path.join(".", output_path)
        with open(output_path, "w") as f:
            f.write(model_card)

        logger.info(f"Model card created: {output_path}")
        return output_path

    def export_model_metadata(
        self, model_metadata: ModelMetadata, output_path: str
    ) -> str:
        """
        Export model metadata to JSON.

        Args:
            model_metadata: Model metadata
            output_path: Path to save metadata

        Returns:
            Path to saved metadata
        """
        metadata_dict = {
            "model_name": model_metadata.model_name,
            "model_path": model_metadata.model_path,
            "base_model": model_metadata.base_model,
            "training_config": model_metadata.training_config,
            "evaluation_metrics": model_metadata.evaluation_metrics,
            "dataset_info": model_metadata.dataset_info,
            "hardware_info": model_metadata.hardware_info,
            "training_time": model_metadata.training_time,
            "model_size": model_metadata.model_size,
            "version": model_metadata.version,
            "export_timestamp": time.time(),
        }

        # Save metadata
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there's a directory component
            os.makedirs(output_dir, exist_ok=True)
        else:
            # If no directory specified, use current directory
            output_path = os.path.join(".", output_path)
        with open(output_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Model metadata exported: {output_path}")
        return output_path

    def log_metrics_wandb(self, metrics: Dict[str, Any], step: int = 0) -> None:
        """Log metrics to Weights & Biases."""
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, skipping metrics logging")
            return

        try:
            wandb.log(metrics, step=step)
            logger.info(f"Logged {len(metrics)} metrics to W&B")
        except Exception as e:
            logger.error(f"Error logging to W&B: {e}")

    def log_model_version_wandb(
        self, model_path: str, model_metadata: ModelMetadata
    ) -> str:
        """Log model version to Weights & Biases."""
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, skipping model logging")
            return ""

        try:
            # Create model artifact
            artifact = wandb.Artifact(
                name=f"{model_metadata.model_name}-v{model_metadata.version}",
                type="model",
                description=f"Fine-tuned model: {model_metadata.model_name}",
            )

            # Add model files
            artifact.add_dir(model_path)

            # Log metadata
            artifact.metadata.update(
                {
                    "base_model": model_metadata.base_model,
                    "training_config": model_metadata.training_config,
                    "evaluation_metrics": model_metadata.evaluation_metrics,
                    "hardware_info": model_metadata.hardware_info,
                    "training_time": model_metadata.training_time,
                    "model_size": model_metadata.model_size,
                }
            )

            # Log the artifact
            wandb.log_artifact(artifact)

            logger.info(f"Logged model version to W&B: {artifact.name}")
            return artifact.name

        except Exception as e:
            logger.error(f"Error logging model to W&B: {e}")
            return ""
