import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

import pandas as pd
from typing import Dict, List, Optional
from collections import Counter
from sklearn.model_selection import train_test_split
from src.preprocess import preprocess

class MultiView_Dataset(Dataset):
    def __init__(self, data, normalizer, mri_stats, pet_stats, view_list):
        super().__init__()
        self.normalizer = normalizer
        self.view_list = view_list
        clinical_df = data.drop(columns=['subject_id', 'visit_id', 'mri_graph_file', 'pet_graph_file', 'label', 'class_label'])
        normalized_data = self.normalizer.transform(clinical_df)
        
        self.clinical_data = torch.tensor(normalized_data, dtype=torch.float32)
        self.mri_data = data[['subject_id', 'visit_id', 'mri_graph_file', 'label']]
        self.pet_data = data[['subject_id', 'visit_id', 'pet_graph_file', 'label']]
        
        self.mri_stats = mri_stats
        self.pet_stats = pet_stats
        
        self.y = torch.tensor(data['label'].to_numpy())
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        label = self.y[idx].item()
        data = []

        for view_id in self.view_list:
            if view_id == 1:
                data.append(self.clinical_data[idx])
            elif view_id == 2:
                X_mri = torch.load(self.mri_data['mri_graph_file'].iloc[idx], weights_only=False)
                X_mri.x = (X_mri.x - self.mri_stats['mean']) / self.mri_stats['std']
                data.append(X_mri)
            elif view_id == 3:
                X_pet = torch.load(self.pet_data['pet_graph_file'].iloc[idx], weights_only=False)
                X_pet.x = (X_pet.x - self.pet_stats['mean']) / self.pet_stats['std']
                data.append(X_pet)
        
        return data, label

def extract_dataset_by_class(df, class_ids: List[int], class_labels: Dict[int, str]):
    
    filtered_df = df[df['label'].isin(class_ids)].copy()
    new_label_map = {old_id: i for i, old_id in enumerate(sorted(class_ids))}
    
    filtered_df['label'] = filtered_df['label'].map(new_label_map)
    
    new_class_labels = {new_id: class_labels[old_id] for old_id, new_id in new_label_map.items()}
    filtered_df['class_label'] = filtered_df['label'].map(new_class_labels)
    
    print(f"Extraction complete. New label mapping: {new_class_labels}")
    return new_class_labels, filtered_df

def subject_wise_split(df, test_size:float=0.2, val_size:float=0.1, random_state:Optional[int]=42):
    
    subjects_df = df.groupby('subject_id')['label'].last().reset_index()

    train_subs, temp_subs = train_test_split(
        subjects_df['subject_id'], 
        test_size=(test_size + val_size), 
        random_state=random_state,
        stratify=subjects_df['label']
    )
    
    relative_val_size = val_size / (test_size + val_size)
    val_subs, test_subs = train_test_split(
        temp_subs, 
        test_size=1-relative_val_size, 
        random_state=random_state
    )
        
    # train_subs_df = subjects_df[subjects_df['subject_id'].isin(train_subs)]
    # min_size = train_subs_df['label'].value_counts().min()
    # train_subs_balanced = (
    #     train_subs_df.groupby('label')
    #     .sample(n=min_size, random_state=random_state)
    #     ['subject_id']
    # )
    # train_df = df[df['subject_id'].isin(train_subs_balanced)]
    
    train_df = df[df['subject_id'].isin(train_subs)]
    val_df   = df[df['subject_id'].isin(val_subs)]
    test_df  = df[df['subject_id'].isin(test_subs)]
    
    print(f'{train_df.label.value_counts()}')

    return train_df, val_df, test_df

def make_balanced_loader(dataset, metadata_df, batch_size:int=16):
    
    labels = metadata_df['label'].values
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                     for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    # weights_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Crucial: allows minority classes to be picked multiple times
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader
        
def load_dataloaders(args):
    dataset = args.dataset
    df = pd.read_csv(f'data/{dataset}/cleaned_merge_df.csv')
    
    class_labels = {0: 'Healthy (CN)', 1: 'MCI', 2: 'Alzheimer\'s (AD)'}
    new_class_labels, subset_df = extract_dataset_by_class(df, class_ids=args.class_ids, class_labels=class_labels)

    train_df, val_df, test_df = subject_wise_split(subset_df)
    normalizer, mri_stats, pet_stats = preprocess(train_df, args.view_list)

    train_dataset = MultiView_Dataset(train_df, normalizer, mri_stats, pet_stats, args.view_list)
    val_dataset   = MultiView_Dataset(val_df, normalizer, mri_stats, pet_stats, args.view_list)
    test_dataset  = MultiView_Dataset(test_df, normalizer, mri_stats, pet_stats, args.view_list)
    
    train_dataloader = make_balanced_loader(train_dataset, train_df, batch_size=args.batch_size)
    val_dataloader   = DataLoader(val_dataset, batch_size=args.batch_size)
    test_dataloader  = DataLoader(test_dataset, batch_size=args.batch_size)
    
    return train_dataloader, val_dataloader, test_dataloader, new_class_labels
