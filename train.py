from datetime import datetime
import argparse
import logging
import sys
import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_utils import load_dataloaders
from src.utils import (
    load_model,
    load_optimizer, 
    Trainer
)

def parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--class-ids', nargs='+', type=int, default=[0, 1, 2],
                        help='Class IDs to include in training.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Mini-batch size for dataloaders.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Maximum number of training epochs.')
    parser.add_argument('--dataset', type=str.lower, choices=['adni', 'aibl', 'oasis'], default='aibl',
                        help='dataset to train on')
    parser.add_argument('--view-list', nargs='+', type=int, default=[1, 2, 3], choices=[1, 2, 3],
                        help='View ids to use for UDD training, where 1=clinical, 2=MRI, 3=PET.')
    parser.add_argument('--hidden-dim', nargs='+', type=int, default=[128, 128, 128],
                        help='Hidden Dimensions for each classifier')
    parser.add_argument('--lambda-epochs', type=int, default=5,
                        help='lambda epoch for UDD training')
    parser.add_argument('--patience', type=int, default=600,
                        help='Early stopping patience in epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--weight-decay', type=float, default=3e-4,
                        help='Weight decay used by the optimizer.')
    parser.add_argument('--logs-dir', type=str, default='logs/',
                        help='Directory for training logs.')
    parser.add_argument('--data-dir', type=str, default='data/preprocessed/',
                        help='Directory containing preprocessed data.')
    parser.add_argument('--save-model-dir', type=str, default='models/',
                        help='Directory where trained models are saved.')
    parser.add_argument('--results-dir', type=str, default='results/',
                        help='Directory where evaluation outputs are saved.')
    
    return parser.parse_args()

def setup_logging(args, timestamp):
    
    
    log_filename = f"logs/{timestamp}.log"
    os.makedirs("logs/", exist_ok=True)
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
    
    return logger


def main():
    
    args = parser()
    args.view_list = list(dict.fromkeys(args.view_list))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = setup_logging(args, timestamp)
    args.save_model_path = f'{args.save_model_dir}/{timestamp}.pth'
    writer = SummaryWriter(f'runs/{timestamp}')
    view_tag = "_".join(str(view_id) for view_id in args.view_list)
    results_dir = os.path.join(args.results_dir, args.dataset, f'views_{view_tag}')
    
    logger.info(f"Starting UDD pipeline with classes {args.class_ids}")
    logger.info("--- Command Line Arguments ---")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info("------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # This needs to be changed maybe we can add a args file for this
    # need a load model function, 
    model = load_model(args)
    model = model.to(device)
    
    train_dataloader, val_dataloader, test_dataloader, class_labels = load_dataloaders(args)
    
    logging.info(model)
    criterion = nn.NLLLoss()
    optimizer = load_optimizer(model, args)
    
    trainer = Trainer(model, 
                      train_dataloader, 
                      val_dataloader, 
                      test_dataloader, 
                      optimizer, 
                      criterion, 
                      writer)
    
    trainer.run(args)
    
    checkpoint = torch.load(args.save_model_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    classification_result, report_dict = trainer.generate_report(class_labels, args, save_path=results_dir)

    logger.info("Classification result:\n%s", classification_result)
    logger.info("Weighted precision: %.4f", report_dict["weighted avg"]["precision"])
    logger.info("Weighted recall: %.4f", report_dict["weighted avg"]["recall"])
    logger.info("Macro F1: %.4f", report_dict["macro avg"]["f1-score"])

    summary_path = trainer.save_results(
        args=args,
        report_dict=report_dict,
        save_path=os.path.join(args.results_dir),
        trained_epochs=checkpoint["epoch"],
    )
    logger.info("Training summary saved to %s", summary_path)
    
    checkpoint['precision'] = report_dict["weighted avg"]["precision"]
    checkpoint['recall'] = report_dict["weighted avg"]["recall"]
    checkpoint['f1_score'] = report_dict["macro avg"]["f1-score"]

    torch.save(checkpoint, args.save_model_path)
    
if __name__ == '__main__':
    main()
