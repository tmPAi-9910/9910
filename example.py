"""
TmpAi Standard 1.0 - Example Usage

This example demonstrates how to initialize, train, evaluate,
and deploy the TmpAi model.
"""

import torch
from tmpai import TmpAiModel
from tmpai.src.training import Trainer
from tmpai.src.evaluation import Evaluator
from tmpai.src.interaction import InteractionProtocol
from tmpai.src.safety import SafetySystem
from tmpai.src.deployment import DeploymentManager, DeploymentConfig


def main():
    """Main example workflow."""
    
    # 1. Initialize Model
    print("Initializing TmpAi Standard 1.0 model...")
    model = TmpAiModel(
        vocab_size=50000,
        embed_dim=4096,
        num_layers=32,
        num_heads=32,
        ff_dim=16384,
        max_seq_len=8192,
        dropout=0.1,
        use_context_retention=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model initialized on {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Setup Training
    print("\nSetting up training...")
    trainer = Trainer(model)
    # trainer.pretrain(train_dataset, eval_dataset)
    
    # 3. Evaluation
    print("\nSetting up evaluation...")
    benchmarks = {
        'perplexity': 15.0,  # Baseline from Claude Opus 4.6
        'bleu': 0.35,
        'rouge': 0.40
    }
    evaluator = Evaluator(model, benchmarks, device)
    # results = evaluator.run_full_evaluation(test_data, save_path='results.json')
    
    # 4. User Interaction
    print("\nSetting up user interaction...")
    interaction = InteractionProtocol(model, max_context_length=4096)
    
    conversation_id = interaction.create_conversation(
        system_prompt="You are TmpAi, a helpful AI assistant."
    )
    
    user_message = "What is artificial intelligence?"
    response = interaction.send_message(conversation_id, user_message)
    print(f"\nUser: {user_message}")
    print(f"Assistant: {response}")
    
    # 5. Safety System
    print("\nInitializing safety system...")
    safety_system = SafetySystem(model)
    
    safety_check = safety_system.check_input(user_message)
    print(f"Input safety check: {safety_check['is_safe']}")
    
    output_check = safety_system.check_output(response)
    print(f"Output safety check: {output_check['is_safe']}")
    
    # 6. Deployment
    print("\nSetting up deployment...")
    config = DeploymentConfig(
        model_path='model/',
        device=str(device),
        api_port=8000
    )
    
    deployment_manager = DeploymentManager(model, config)
    # deployment_manager.prepare_deployment('deployment/', formats=['pytorch', 'torchscript'])
    # deployment_manager.deploy_docker()
    
    print("\nTmpAi Standard 1.0 example completed successfully!")


if __name__ == '__main__':
    main()
