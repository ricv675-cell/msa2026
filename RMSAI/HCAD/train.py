"""
H²-CAD Training Script.
层次化双曲确定性感知蒸馏训练脚本

Implements the full training pipeline for H²-CAD model including:
- EMA (Exponential Moving Average) Teacher for knowledge distillation
- Poincaré ball embeddings for hierarchical uncertainty modeling
- Completeness-aware fusion with Cross-Modal Compensation
- Progressive hyperbolic distillation scheduling
- Early stopping

Reference:
    Hierarchical Hyperbolic Certainty-Aware Distillation for
    Multimodal Sentiment Analysis with Incomplete Data
"""

import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import H2CADLoss
from core.scheduler import get_scheduler, get_hyperbolic_scheduler
from core.utils import setup_seed, get_best_results
from core.ema import EMATeacher
from models.h2cad import build_model
from core.metric import MetricsTop


# Device setup
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(f"Device: {device}")

# Argument parser
parser = argparse.ArgumentParser(description='H²-CAD Training')
parser.add_argument('--config_file', type=str, default='',
                    help='Path to configuration YAML file')
parser.add_argument('--seed', type=int, default=-1,
                    help='Random seed (-1 to use config seed)')
opt = parser.parse_args()
print(opt)


def train_epoch(model, ema_teacher, train_loader, optimizer, loss_fn,
                distillation_scheduler, epoch):
    """
    Train H²-CAD for one epoch.

    Args:
        model: H²-CAD student model
        ema_teacher: EMA teacher for certainty-aware knowledge distillation
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: H²-CAD composite loss function
        distillation_scheduler: Progressive distillation weight scheduler β(t)
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
        Current distillation weight β(t)
    """
    model.train()
    beta_weight = distillation_scheduler.get_weight(epoch)

    y_pred, y_true = [], []
    loss_dict = {}
    total_loss = 0.0

    for cur_iter, data in enumerate(train_loader):
        # Prepare complete inputs (for teacher)
        complete_input = (
            data['vision'].to(device),
            data['audio'].to(device),
            data['text'].to(device)
        )
        # Prepare incomplete inputs (for student)
        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device)
        )

        # Prepare labels
        sentiment_labels = data['labels']['M'].to(device)
        # Completeness label: ŵ* = 1 - missing_rate
        completeness_labels = 1. - data['labels']['missing_rate_l'].to(device)

        # ===== Forward pass - Student =====
        optimizer.zero_grad()
        student_out = model(complete_input, incomplete_input, return_teacher=False)

        # ===== Forward pass - EMA Teacher (no gradient) =====
        with torch.no_grad():
            teacher_out = ema_teacher.model(complete_input, complete_input, return_teacher=False)
            teacher_z = teacher_out['z_hyp'].detach()

        # ===== Compute H²-CAD Loss =====
        # L = L_task + λ₁·L_comp + β(t)·L_hyp + λ₂·L_recon
        loss_result = loss_fn(
            student_out=student_out,
            teacher_z=teacher_z,
            labels=sentiment_labels,
            completeness_labels=completeness_labels,
            beta_weight=beta_weight
        )

        # Backward pass
        loss = loss_result['loss']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update student model
        optimizer.step()

        # Update EMA teacher (after optimizer step)
        ema_teacher.update(model)

        # Collect predictions
        y_pred.append(student_out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())

        # Accumulate loss
        total_loss += loss.item()

        # Accumulate loss components
        if cur_iter == 0:
            for key, value in loss_result.items():
                if key != 'loss':
                    loss_dict[key] = value
        else:
            for key, value in loss_result.items():
                if key != 'loss':
                    loss_dict[key] += value

    # Average loss components
    n_batches = len(train_loader)
    loss_dict = {key: value / n_batches for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict} | β(t)={beta_weight:.4f}')

    return total_loss / n_batches, beta_weight


@torch.no_grad()
def evaluate(model, eval_loader, metrics):
    """
    Evaluate H²-CAD model on validation/test set.

    During evaluation, only incomplete inputs are used to simulate
    real-world scenarios with missing modalities.

    Args:
        model: H²-CAD model to evaluate
        eval_loader: Evaluation data loader
        metrics: Metrics function

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    y_pred, y_true = [], []

    for data in eval_loader:
        # Use only incomplete input during evaluation
        incomplete_input = (
            data['vision_m'].to(device),
            data['audio_m'].to(device),
            data['text_m'].to(device)
        )

        sentiment_labels = data['labels']['M'].to(device)

        # Forward pass with complete_input as None
        out = model((None, None, None), incomplete_input, return_teacher=False)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())

    pred = torch.cat(y_pred)
    true = torch.cat(y_true)

    return metrics(pred, true)


def main():
    """
    Main training function for H²-CAD.

    Implements the full training pipeline:
    1. Load configuration and setup
    2. Build H²-CAD model and EMA Teacher
    3. Training loop with progressive distillation
    4. Validation and early stopping
    """
    best_valid_results, best_test_results = {}, {}

    # Load configuration
    config_file = 'configs/train_sims.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    # Set seed
    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print(f"Seed is fixed to {seed}")

    # Create checkpoint directory
    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print(f"Checkpoint root: {ckpt_root}")

    # ===== Build H²-CAD Model =====
    model = build_model(args).to(device)
    print(f"H²-CAD model built with {sum(p.numel() for p in model.parameters()):,} parameters")

    # ===== Create EMA Teacher =====
    ema_decay = args['base'].get('ema_decay', 0.999)
    ema_teacher = EMATeacher(model, decay=ema_decay)
    print(f"EMA Teacher initialized with decay={ema_decay}")

    # Data loader
    dataLoader = MMDataLoader(args)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args['base']['lr'],
        weight_decay=args['base']['weight_decay']
    )

    # ===== Schedulers =====
    scheduler_warmup = get_scheduler(optimizer, args)
    # Progressive distillation scheduler: β(t)
    distillation_scheduler = get_hyperbolic_scheduler(args)
    print(f"Progressive distillation: warmup={args['base'].get('hyp_warmup_epochs', 20)} epochs")

    # ===== H²-CAD Loss Function =====
    loss_fn = H2CADLoss(args)
    print(f"Loss weights: α={loss_fn.alpha}, γ={loss_fn.gamma}")

    # Metrics
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(args['dataset']['datasetName'])

    # Early stopping setup
    early_stop_patience = args['base'].get('early_stop_patience', None)
    best_mae = float('inf')
    no_improve = 0

    # ===== Training Loop =====
    print("\n" + "=" * 50)
    print("Starting H²-CAD Training")
    print("=" * 50 + "\n")

    for epoch in range(1, args['base']['n_epochs'] + 1):
        # Train
        train_loss, beta_weight = train_epoch(
            model, ema_teacher, dataLoader['train'],
            optimizer, loss_fn, distillation_scheduler, epoch
        )

        # Validation
        if args['base'].get('do_validation', True):
            valid_results = evaluate(model, dataLoader['valid'], metrics)
            best_valid_results = get_best_results(
                valid_results, best_valid_results, epoch,
                model, optimizer, ckpt_root, seed,
                save_best_model=args['base'].get('save_best_model', False)
            )
            print(f'Valid Epoch {epoch}: {valid_results}')
            print(f'Current Best Valid Results: {best_valid_results}')

        # Test
        test_results = evaluate(model, dataLoader['test'], metrics)
        best_test_results = get_best_results(
            test_results, best_test_results, epoch,
            model, optimizer, ckpt_root, seed,
            save_best_model=args['base'].get('save_best_model', True)
        )
        print(f'Test Epoch {epoch}: {test_results}')
        print(f'Current Best Test Results: {best_test_results}\n')

        # Update learning rate
        scheduler_warmup.step()

        # Early stopping check
        if early_stop_patience is not None:
            current_results = valid_results if args['base'].get('do_validation', True) else test_results
            current_mae = current_results.get('MAE', None)

            if current_mae is not None:
                if current_mae < best_mae - 1e-4:
                    best_mae = current_mae
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= early_stop_patience:
                        print(f"Early stopping at epoch {epoch} (no MAE improvement for {early_stop_patience} epochs).")
                        break

    print("\n" + "=" * 50)
    print("H²-CAD Training completed!")
    print(f"Best Valid Results: {best_valid_results}")
    print(f"Best Test Results: {best_test_results}")


if __name__ == '__main__':
    main()
