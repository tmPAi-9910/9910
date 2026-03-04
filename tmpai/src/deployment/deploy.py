"""
TmpAi Standard 1.0 - Deployment Module
"""

import torch
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
import yaml
from datetime import datetime
import subprocess
import shutil
from dataclasses import dataclass, field

from tmpai.models import TmpAiModel


@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    model_path: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 1
    max_sequence_length: int = 8192
    temperature: float = 0.7
    top_p: float = 0.9
    enable_cache: bool = True
    enable_streaming: bool = True
    api_enabled: bool = True
    api_port: int = 8000
    auth_enabled: bool = True
    rate_limit: int = 100  # requests per minute
    log_level: str = 'INFO'


class ModelExporter:
    """Exports model for deployment in various formats."""
    
    def __init__(self, model: TmpAiModel):
        self.model = model
    
    def export_pytorch(self, output_path: str, include_optimizer: bool = False) -> None:
        """
        Export model in PyTorch format.
        
        Args:
            output_path: Path to save the model
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = {
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embed_dim': self.model.embed_dim,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'max_seq_len': self.model.max_seq_len
            },
            'model_state_dict': self.model.state_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, output_path)
        print(f"Model exported to PyTorch format: {output_path}")
    
    def export_onnx(self, output_path: str, opset_version: int = 14) -> None:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
        """
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randint(0, self.model.vocab_size, (1, 128))
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"Model exported to ONNX format: {output_path}")
    
    def export_torchscript(self, output_path: str) -> None:
        """
        Export model to TorchScript for deployment.
        
        Args:
            output_path: Path to save TorchScript model
        """
        self.model.eval()
        
        # Create example input
        example_input = torch.randint(0, self.model.vocab_size, (1, 128))
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, example_input)
        
        # Save traced model
        traced_model.save(output_path)
        
        print(f"Model exported to TorchScript format: {output_path}")
    
    def quantize_model(self, output_path: str, dtype: str = 'int8') -> None:
        """
        Quantize model for reduced memory footprint.
        
        Args:
            output_path: Path to save quantized model
            dtype: Target dtype ('int8' or 'float16')
        """
        if dtype == 'int8':
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
        elif dtype == 'float16':
            quantized_model = self.model.half()
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        torch.save(quantized_model.state_dict(), output_path)
        print(f"Model quantized to {dtype}: {output_path}")


class DockerBuilder:
    """Builds Docker containers for deployment."""
    
    def __init__(self, model_path: str, config: DeploymentConfig):
        self.model_path = model_path
        self.config = config
    
    def generate_dockerfile(self, output_path: str = 'Dockerfile') -> None:
        """Generate Dockerfile for model deployment."""
        dockerfile_content = f"""
# TmpAi Standard 1.0 Deployment
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY {self.model_path} /app/model/
COPY tmpai/ /app/tmpai/

# Copy configuration
COPY config/ /app/config/

# Expose API port
EXPOSE {self.config.api_port}

# Set environment variables
ENV DEVICE={self.config.device}
ENV BATCH_SIZE={self.config.batch_size}
ENV MAX_SEQUENCE_LENGTH={self.config.max_sequence_length}
ENV API_PORT={self.config.api_port}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:{self.config.api_port}/health || exit 1

# Run the API server
CMD ["python", "-m", "tmpai.deployment.api_server"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content.strip())
        
        print(f"Dockerfile generated: {output_path}")
    
    def generate_docker_compose(self, output_path: str = 'docker-compose.yml') -> None:
        """Generate docker-compose.yml for multi-service deployment."""
        compose_content = f"""
version: '3.8'

services:
  tmpai-api:
    build: .
    container_name: tmpai-api
    ports:
      - "{self.config.api_port}:{self.config.api_port}"
    environment:
      - DEVICE={self.config.device}
      - BATCH_SIZE={self.config.batch_size}
      - API_PORT={self.config.api_port}
    volumes:
      - ./model:/app/model
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:{self.config.api_port}/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  tmpai-worker:
    build: .
    command: python -m tmpai.deployment.worker
    environment:
      - WORKER_TYPE=training
    volumes:
      - ./model:/app/model
      - ./data:/app/data
    restart: unless-stopped
  
  redis:
    image: redis:7-alpine
    container_name: tmpai-redis
    ports:
      - "6379:6379"
    restart: unless-stopped
  
  prometheus:
    image: prom/prometheus:latest
    container_name: tmpai-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
  
  grafana:
    image: grafana/grafana:latest
    container_name: tmpai-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana:/var/lib/grafana
    restart: unless-stopped
"""
        
        with open(output_path, 'w') as f:
            f.write(compose_content.strip())
        
        print(f"docker-compose.yml generated: {output_path}")
    
    def build_image(self, tag: str = 'tmpai-standard:1.0') -> None:
        """Build Docker image."""
        self.generate_dockerfile()
        
        subprocess.run(
            ['docker', 'build', '-t', tag, '.'],
            check=True
        )
        
        print(f"Docker image built: {tag}")


class KubernetesDeployer:
    """Generates Kubernetes manifests for deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_deployment(self, output_path: str = 'k8s/deployment.yaml') -> None:
        """Generate Kubernetes deployment manifest."""
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'tmpai-api',
                'labels': {
                    'app': 'tmpai',
                    'component': 'api'
                }
            },
            'spec': {
                'replicas': 3,
                'selector': {
                    'matchLabels': {
                        'app': 'tmpai',
                        'component': 'api'
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'tmpai',
                            'component': 'api'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'tmpai-api',
                            'image': 'tmpai-standard:1.0',
                            'ports': [{
                                'containerPort': self.config.api_port
                            }],
                            'env': [
                                {'name': 'DEVICE', 'value': self.config.device},
                                {'name': 'API_PORT', 'value': str(self.config.api_port)},
                                {'name': 'BATCH_SIZE', 'value': str(self.config.batch_size)}
                            ],
                            'resources': {
                                'requests': {
                                    'memory': '8Gi',
                                    'cpu': '4'
                                },
                                'limits': {
                                    'memory': '16Gi',
                                    'cpu': '8'
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(deployment, f, default_flow_style=False)
        
        print(f"Kubernetes deployment manifest: {output_path}")
    
    def generate_service(self, output_path: str = 'k8s/service.yaml') -> None:
        """Generate Kubernetes service manifest."""
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'tmpai-api'
            },
            'spec': {
                'selector': {
                    'app': 'tmpai',
                    'component': 'api'
                },
                'ports': [{
                    'port': 80,
                    'targetPort': self.config.api_port,
                    'protocol': 'TCP'
                }],
                'type': 'LoadBalancer'
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(service, f, default_flow_style=False)
        
        print(f"Kubernetes service manifest: {output_path}")
    
    def generate_hpa(self, output_path: str = 'k8s/hpa.yaml') -> None:
        """Generate HorizontalPodAutoscaler manifest."""
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': 'tmpai-api-hpa'
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': 'tmpai-api'
                },
                'minReplicas': 2,
                'maxReplicas': 10,
                'metrics': [{
                    'type': 'Resource',
                    'resource': {
                        'name': 'cpu',
                        'target': {
                            'type': 'Utilization',
                            'averageUtilization': 70
                        }
                    }
                }]
            }
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(hpa, f, default_flow_style=False)
        
        print(f"Kubernetes HPA manifest: {output_path}")


class ContinuousLearning:
    """
    Manages continuous learning and model updates based on
    real-world usage and user feedback.
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        feedback_path: str = 'feedback/',
        checkpoint_dir: str = 'checkpoints/'
    ):
        self.model = model
        self.feedback_path = Path(feedback_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.version = 1
        self.update_log: List[Dict[str, Any]] = []
    
    def collect_feedback(self) -> List[Dict[str, Any]]:
        """Collect all available feedback data."""
        feedback_data = []
        
        for feedback_file in self.feedback_path.glob('*.json'):
            with open(feedback_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    feedback_data.extend(data)
                else:
                    feedback_data.append(data)
        
        return feedback_data
    
    def analyze_feedback(self, feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze feedback to identify improvement areas.
        
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'total_feedback': len(feedback),
            'average_rating': 0.0,
            'common_issues': [],
            'improvement_areas': [],
            'positive_aspects': []
        }
        
        if not feedback:
            return analysis
        
        # Calculate average rating
        ratings = [f['feedback_value'] for f in feedback 
                   if f['feedback_type'] == 'rating']
        if ratings:
            analysis['average_rating'] = sum(ratings) / len(ratings)
        
        # Count issue types
        issue_counts = {}
        for f in feedback:
            if f['feedback_type'] == 'flag':
                issue = f.get('comment', 'No comment')
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Get most common issues
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        analysis['common_issues'] = [issue[0] for issue in sorted_issues[:5]]
        
        # Identify improvement areas based on feedback
        if analysis['average_rating'] < 3.0:
            analysis['improvement_areas'].append('Response quality')
        if 'accuracy' in str(issue_counts).lower():
            analysis['improvement_areas'].append('Factuality')
        if 'relevance' in str(issue_counts).lower():
            analysis['improvement_areas'].append('Relevance')
        
        return analysis
    
    def schedule_update(
        self,
        improvement_threshold: float = 0.8,
        min_feedback_count: int = 100
    ) -> bool:
        """
        Determine if a model update is warranted.
        
        Returns:
            True if update should be scheduled
        """
        feedback = self.collect_feedback()
        
        if len(feedback) < min_feedback_count:
            print(f"Not enough feedback for update ({len(feedback)}/{min_feedback_count})")
            return False
        
        analysis = self.analyze_feedback(feedback)
        
        # Check if average rating is below threshold
        if analysis['average_rating'] < improvement_threshold:
            print(f"Average rating ({analysis['average_rating']:.2f}) below threshold "
                  f"({improvement_threshold}) - update warranted")
            return True
        
        return False
    
    def create_update(self, description: str) -> str:
        """
        Create a new model version.
        
        Returns:
            Version identifier
        """
        self.version += 1
        version_id = f"v{self.version}.{datetime.now().strftime('%Y%m%d')}"
        
        checkpoint_path = self.checkpoint_dir / f"model_{version_id}.pt"
        torch.save({
            'version': version_id,
            'model_state_dict': self.model.state_dict(),
            'description': description,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        # Log update
        self.update_log.append({
            'version': version_id,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'checkpoint_path': str(checkpoint_path)
        })
        
        print(f"Created model update: {version_id}")
        return version_id
    
    def rollback_to_version(self, version_id: str) -> None:
        """Rollback model to a previous version."""
        checkpoint_path = self.checkpoint_dir / f"model_{version_id}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Version {version_id} not found")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Rolled back to version {version_id}")
    
    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get history of model updates."""
        return self.update_log


class DeploymentManager:
    """
    Main deployment manager for TmpAi Standard 1.0
    
    Orchestrates model export, containerization, and deployment
    across different platforms.
    """
    
    def __init__(
        self,
        model: TmpAiModel,
        config: Optional[DeploymentConfig] = None
    ):
        self.model = model
        self.config = config or DeploymentConfig(model_path='model/')
        self.exporter = ModelExporter(model)
        self.continuous_learning = ContinuousLearning(model)
    
    def prepare_deployment(
        self,
        output_dir: str = 'deployment/',
        formats: List[str] = ['pytorch', 'torchscript']
    ) -> None:
        """
        Prepare model for deployment in specified formats.
        
        Args:
            output_dir: Directory to save deployment artifacts
            formats: List of export formats
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export in requested formats
        if 'pytorch' in formats:
            self.exporter.export_pytorch(
                str(output_path / 'model_pytorch.pt'),
                include_optimizer=False
            )
        
        if 'onnx' in formats:
            self.exporter.export_onnx(str(output_path / 'model.onnx'))
        
        if 'torchscript' in formats:
            self.exporter.export_torchscript(str(output_path / 'model_torchscript.pt'))
        
        # Generate quantized version
        self.exporter.quantize_model(
            str(output_path / 'model_quantized_int8.pt'),
            dtype='int8'
        )
        
        print(f"Deployment artifacts prepared in {output_dir}")
    
    def deploy_docker(self, tag: str = 'tmpai-standard:1.0') -> None:
        """Deploy model using Docker."""
        docker_builder = DockerBuilder('model/', self.config)
        docker_builder.generate_dockerfile()
        docker_builder.generate_docker_compose()
        docker_builder.build_image(tag)
        
        print("Docker deployment ready. Use 'docker-compose up' to start services.")
    
    def deploy_kubernetes(self) -> None:
        """Deploy model to Kubernetes."""
        k8s_deployer = KubernetesDeployer(self.config)
        k8s_path = Path('k8s')
        k8s_path.mkdir(parents=True, exist_ok=True)
        
        k8s_deployer.generate_deployment(str(k8s_path / 'deployment.yaml'))
        k8s_deployer.generate_service(str(k8s_path / 'service.yaml'))
        k8s_deployer.generate_hpa(str(k8s_path / 'hpa.yaml'))
        
        print("Kubernetes manifests generated. Use 'kubectl apply -f k8s/' to deploy.")
    
    def setup_continuous_learning(
        self,
        check_interval_hours: int = 24
    ) -> None:
        """
        Setup continuous learning framework.
        
        Args:
            check_interval_hours: Hours between update checks
        """
        self.continuous_learning.schedule_update()
        
        print(f"Continuous learning enabled (check interval: {check_interval_hours} hours)")
        print("Monitor feedback/ directory for user data")
    
    def generate_deployment_guide(self, output_path: str = 'DEPLOYMENT.md') -> None:
        """Generate deployment guide documentation."""
        guide = f"""
# TmpAi Standard 1.0 - Deployment Guide

## Quick Start

### Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t tmpai-standard:1.0 .
   ```

2. Start the service:
   ```bash
   docker-compose up -d
   ```

3. Check health:
   ```bash
   curl http://localhost:{self.config.api_port}/health
   ```

### Kubernetes Deployment

1. Apply manifests:
   ```bash
   kubectl apply -f k8s/
   ```

2. Check status:
   ```bash
   kubectl get pods
   kubectl get svc
   ```

## Configuration

Environment variables:
- `DEVICE`: Device to use (cuda/cpu)
- `BATCH_SIZE`: Batch size for inference
- `MAX_SEQUENCE_LENGTH`: Maximum sequence length
- `API_PORT`: API server port

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Continuous Learning

Model updates are automatically scheduled based on user feedback.
Check `feedback/` directory for collected feedback data.
"""
        
        with open(output_path, 'w') as f:
            f.write(guide.strip())
        
        print(f"Deployment guide generated: {output_path}")
