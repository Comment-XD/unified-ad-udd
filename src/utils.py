from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import logging
import os

import torch
import torch.nn as nn
from torch_geometric.data import Data


from typing import List
from src.model import GNNClassifier, TabularClassifer, UDD
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_model(args) -> torch.nn.Module:
    n_classes = len(args.class_ids)
    # print(n_classes)
    hidden_dim = args.hidden_dim
    if len(hidden_dim) < 3:
        raise ValueError(f"Expected 3 hidden dimensions for clinical/MRI/PET, got {len(hidden_dim)}")
    
    classifiers = {
        'adni': [
            TabularClassifer(input_dim=22, hidden_dim=hidden_dim[0], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=113, hidden_dim=hidden_dim[1], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=5, hidden_dim=hidden_dim[2], num_classes=n_classes, softplus=True)
        ],
        
        'aibl': [
            TabularClassifer(input_dim=26, hidden_dim=hidden_dim[0], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=113, hidden_dim=hidden_dim[1], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=5, hidden_dim=hidden_dim[2], num_classes=n_classes, softplus=True)
        ],
        
        'oasis': [
            TabularClassifer(input_dim=48, hidden_dim=hidden_dim[0], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=113, hidden_dim=hidden_dim[1], num_classes=n_classes, softplus=True),
            GNNClassifier(node_feature_dim=7, hidden_dim=hidden_dim[2], num_classes=n_classes, softplus=True)
        ]}

    selected_classifiers = nn.ModuleList(
        classifiers[args.dataset][view_id - 1] for view_id in args.view_list
    )
    
    model = UDD(selected_classifiers,
                num_classes=n_classes, 
                lambda_epochs=args.lambda_epochs)
    
    return model

def save_model(model, optimizer, epoch, val_acc, args):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': vars(args)
            }, args.save_model_path)


def load_optimizer(model, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optimizer
    
class Trainer(object):
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_dataloader: torch.utils.data.DataLoader, 
                 val_dataloader: torch.utils.data.DataLoader, 
                 test_dataloader: torch.utils.data.DataLoader, 
                 optimizer, 
                 criterion, 
                 writer):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.writer = writer
        
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []
        
    def _move_views_to_device(self, data):
        views = list(data)
        for v_num in range(len(views)):
            views[v_num] = views[v_num].to(device)
        return views
        
    def train_one_epoch(self, epoch: int):
        self.model.train()
        
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        for _, (data, target) in enumerate(self.train_dataloader):
            data = self._move_views_to_device(data)
            target = target.long().to(device)
            self.optimizer.zero_grad()
            evidences, evidence_a, loss, u_a = self.model(data, target, epoch)
            pred = evidence_a.argmax(dim=1)
            acc = (target == pred).sum() * 100 / len(target)
            
            # Add accuracy for evidence to log and 
            loss.backward()
            self.optimizer.step()
            
            acc_meter.update(acc.item())
            loss_meter.update(loss.item())
        
        return loss_meter.avg, acc_meter.avg
    
    def val_one_epoch(self, epoch: int):
        self.model.eval()
    
        with torch.no_grad():
            loss_meter = AverageMeter()
            acc_meter = AverageMeter()
            
            for _, (data, target) in enumerate(self.val_dataloader):
                data = self._move_views_to_device(data)
                target = target.long().to(device)
                evidences, evidence_a, loss, u_a = self.model(data, target, epoch)
                pred = evidence_a.argmax(dim=1)
                
                acc = (target == pred).sum() * 100 / len(target)
                
                acc_meter.update(acc.item())
                loss_meter.update(loss.item())
    
        return loss_meter.avg, acc_meter.avg
    
    def run(self, args):

        best_val_acc = 0
        patience_counter = 0
        self.writer.add_text('Config/Classes', str(args.class_ids))
        for epoch in range(1, args.epochs + 1):
            
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.val_one_epoch(epoch)
            logging.info(f'E{epoch}| Train Loss - {train_loss : .4f} - Train Accuracy - {train_acc : .3f}% \
Val Loss - {val_loss : .3f} - Val Accuracy - {val_acc : .4f}%')
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                save_model(self.model, self.optimizer, epoch, val_acc, args)
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print("Early stopping!")
                break
            

        self.writer.close()
        
    def save_results(self, args, report_dict, save_path: str, trained_epochs: int):
        os.makedirs(save_path, exist_ok=True)
        summary_path = os.path.join(save_path, "training_summary.csv")

        available_views = [1, 2, 3]
        available_classes = [0, 1, 2]

        summary_row = {
            'dataset': args.dataset,
            'epochs': trained_epochs,
            'hidden_dim_1': args.hidden_dim[0],
            'hidden_dim_2': args.hidden_dim[1],
            'hidden_dim_3': args.hidden_dim[2],
            'view_1': int(1 in args.view_list),
            'view_2': int(2 in args.view_list),
            'view_3': int(3 in args.view_list),
            'class_0': int(0 in args.class_ids),
            'class_1': int(1 in args.class_ids),
            'class_2': int(2 in args.class_ids),
            'weighted_accuracy_score': report_dict.get("weighted avg", {}).get("recall", report_dict.get("accuracy")),
            'avg_uncertainty': report_dict['avg_uncertainty'],
            'auc_score': report_dict.get("auc", np.nan),
            'macro_f1_score': report_dict.get("macro avg", {}).get("f1-score", np.nan),
            'model_save_path': args.save_model_path
        }

        ordered_columns = (
            ["dataset", "epochs"]
            + [f"view_{view_id}" for view_id in available_views]
            + [f"class_{class_id}" for class_id in available_classes]
            + [f'hidden_dim_{i + 1}' for i in range(3)]
            + ["avg_uncertainty", "weighted_accuracy_score", "auc_score", "macro_f1_score"]
            + ['model_save_path']
        )

        summary_df = pd.DataFrame([summary_row], columns=ordered_columns)
        if os.path.exists(summary_path):
            summary_df.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)

        return summary_path
    
    def generate_report(self, class_labels, args, save_path: str = "results"):
        classes = len(class_labels)
        hidden_size_str = "_".join(map(str, args.hidden_dim))
        class_id_str = "_".join(map(str, args.class_ids))# 
        os.makedirs(f'{save_path}/{hidden_size_str}', exist_ok=True)

        self.model.eval()
        y_true, y_pred, y_prob, uncertainties = [], [], [], []

        with torch.no_grad():
            for data, target in self.test_dataloader:

                data = self._move_views_to_device(data)
                target = target.long().to(device)


                _, evidence_a, _, u_a = self.model(data, target, args.epochs)
                pred = evidence_a.argmax(dim=1)
                
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())
                uncertainties.extend(u_a.cpu().numpy().flatten())
                
                prob = torch.softmax(evidence_a, dim=1)
                y_prob.extend(prob.cpu().numpy())

        target_names = [class_labels[idx] for idx in sorted(class_labels)]
        results = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'u': uncertainties})
        results.to_csv(f'{save_path}/{hidden_size_str}/{class_id_str}_results.csv')
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary' if classes == 2 else 'macro')
        avg_u = np.mean(uncertainties)
        report_dict = classification_report(
            y_true,
            y_pred,
            labels=sorted(class_labels),
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        classification_result = (
            f"Accuracy: {acc:.4f}\n"
            f"F1-Score: {f1:.4f}\n"
            f"Avg Uncertainty: {avg_u:.4f}"
        )

        report_dict["accuracy"] = acc
        report_dict["avg_uncertainty"] = avg_u

        if len(y_prob) > 0:
            try:
                y_prob_array = np.asarray(y_prob)
                if classes == 2:
                    auc = roc_auc_score(y_true, y_prob_array[:, 1])
                else:
                    auc = roc_auc_score(
                        y_true,
                        y_prob_array,
                        multi_class="ovr",
                        average="weighted",
                    )
                print(f"AUC: {auc:.4f}")
                report_dict["auc"] = auc
                classification_result += f"\nAUC: {auc:.4f}"
            except ValueError:
                report_dict["auc"] = np.nan

        return classification_result, report_dict


# def run_aibl_staged_threshold_grid_search(save_root: str, num_thresholds: int = 100):
#     view_paths = {
#         1: os.path.join(save_root, "views_1", "test_predictions.csv"),
#         2: os.path.join(save_root, "views_2", "test_predictions.csv"),
#         3: os.path.join(save_root, "views_3", "test_predictions.csv"),
#     }
    
#     # Checks for each save view path, checks to see if those test predictions are there
    

#     missing_paths = [path for path in view_paths.values() if not os.path.exists(path)]
#     if missing_paths:
#         raise FileNotFoundError(
#             "Missing staged-threshold inputs: "
#             + ", ".join(missing_paths)
#             + ". Run AIBL evaluation for views=1,2,3 first."
#         )


#     # reads the results for each
#     df_uni = pd.read_csv(view_paths[1]).copy()
#     df_bi = pd.read_csv(view_paths[2]).copy()
#     df_tri = pd.read_csv(view_paths[3]).copy()

#     required_columns = {"patient_id", "u", "pred", "label"}
#     for name, df in [("views_1", df_uni), ("views_2", df_bi), ("views_3", df_tri)]:
#         missing_columns = required_columns - set(df.columns)
#         if missing_columns:
#             raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")

#     df_uni = df_uni.sort_values("patient_id").reset_index(drop=True)
#     df_bi = df_bi.sort_values("patient_id").reset_index(drop=True)
#     df_tri = df_tri.sort_values("patient_id").reset_index(drop=True)

#     patient_ids = df_uni["patient_id"].tolist()
#     if patient_ids != df_bi["patient_id"].tolist() or patient_ids != df_tri["patient_id"].tolist():
#         raise ValueError("patient_id order must match across AIBL views_1, views_2, and views_3 results")

#     threshold_range_uni = np.linspace(df_uni["u"].min(), df_uni["u"].max(), num_thresholds)
#     threshold_range_bi = np.linspace(df_bi["u"].min(), df_bi["u"].max(), num_thresholds)

#     best_accuracy = -1.0
#     best_threshold_t1 = None
#     best_threshold_t2 = None
#     best_stage_df = None

#     for t1 in threshold_range_uni:
#         df_uni_confident = df_uni[df_uni["u"] <= t1].copy()
#         df_uni_confident.loc[:, "stage"] = "views_1"

#         remaining_after_uni = df_uni.loc[~df_uni["patient_id"].isin(df_uni_confident["patient_id"])]
#         df_staged = df_uni_confident[["patient_id", "stage", "u", "pred", "label"]].copy()

#         for t2 in threshold_range_bi:
#             candidate_df = df_staged.copy()

#             if not remaining_after_uni.empty:
#                 df_bi_sub = df_bi[df_bi["patient_id"].isin(remaining_after_uni["patient_id"])].copy()
#                 df_bi_confident = df_bi_sub[df_bi_sub["u"] <= t2].copy()
#                 df_bi_confident.loc[:, "stage"] = "views_2"
#                 candidate_df = pd.concat(
#                     [candidate_df, df_bi_confident[["patient_id", "stage", "u", "pred", "label"]]],
#                     ignore_index=True,
#                 )

#                 remaining_after_bi = df_bi_sub.loc[~df_bi_sub["patient_id"].isin(df_bi_confident["patient_id"])]
#                 if not remaining_after_bi.empty:
#                     df_tri_sub = df_tri[df_tri["patient_id"].isin(remaining_after_bi["patient_id"])].copy()
#                     df_tri_sub.loc[:, "stage"] = "views_3"
#                     candidate_df = pd.concat(
#                         [candidate_df, df_tri_sub[["patient_id", "stage", "u", "pred", "label"]]],
#                         ignore_index=True,
#                     )

#             candidate_df = candidate_df.sort_values("patient_id").reset_index(drop=True)
#             if candidate_df.shape[0] != df_uni.shape[0]:
#                 continue

#             total_accuracy = accuracy_score(candidate_df["label"], candidate_df["pred"])
#             if total_accuracy > best_accuracy:
#                 best_accuracy = total_accuracy
#                 best_threshold_t1 = float(t1)
#                 best_threshold_t2 = float(t2)
#                 best_stage_df = candidate_df.copy()

#     if best_stage_df is None:
#         raise RuntimeError("AIBL staged threshold search did not produce a complete prediction table")

#     staged_results_path = os.path.join(save_root, "staged_results.csv")
#     staged_metrics_path = os.path.join(save_root, "staged_performance.csv")
#     best_stage_df.to_csv(staged_results_path, index=False)

#     staged_f1 = f1_score(best_stage_df["label"], best_stage_df["pred"], average="weighted")
#     staged_report = {
#         "dataset": "AIBL",
#         "acc": best_accuracy,
#         "measure1": staged_f1,
#         "measure2": np.nan,
#         "t1": best_threshold_t1,
#         "t2": best_threshold_t2,
#     }
#     pd.DataFrame([staged_report]).to_csv(staged_metrics_path, index=False)

#     return staged_results_path, staged_metrics_path, staged_report
