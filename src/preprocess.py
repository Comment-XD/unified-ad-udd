from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

from typing import Dict, List, Optional, Tuple

def compute_graph_normalized_stats(data: pd.Series) -> Dict[str, float]:
    '''
    data (pd.Series): Pandas Series containing the graph paths 
    
    Return 
    (Dict[str, float]): 
    
    The dictionary containing the mean and std for z-score normalization for the graphical features
    '''
    
    total_sum = 0
    total_sum_squared = 0
    total_nodes = 0
    
    for graph_path in data:
        graph = torch.load(graph_path, weights_only=False)
        x = graph.x.float()
        
        total_sum += x.sum(dim=0)
        total_sum_squared += (x ** 2).sum(dim=0)
        total_nodes += x.size(0)

    mean = total_sum / total_nodes
    var = total_sum_squared / total_nodes - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    return {'mean':mean.float(), 'std':std.float()}
    
def preprocess(data: pd.DataFrame, view_list: Optional[List[int]] = None) -> Tuple[StandardScaler, Optional[Dict], Optional[Dict]]:
    '''
    data (pd.DataFrame): the train dataset used to calculate the clinical normalizer and MRI
    and PET normalization parameters (mean and standard deviation)
    
    Return (Tuple[StandardScaler, Dict, Dict])
    '''
    
    clinical_df = data.drop(columns=['subject_id', 'visit_id', 'mri_graph_file', 'pet_graph_file', 'label', 'class_label'])
    clinical_normalizer = StandardScaler()
    selected_views = set(view_list or [1, 2, 3])
    
    mri_data = data[['subject_id', 'visit_id', 'mri_graph_file', 'label']]
    pet_data = data[['subject_id', 'visit_id', 'pet_graph_file', 'label']]
    
    clinical_normalizer.fit(clinical_df)
    
    mri_stats = compute_graph_normalized_stats(mri_data['mri_graph_file']) if 2 in selected_views else None
    pet_stats = compute_graph_normalized_stats(pet_data['pet_graph_file']) if 3 in selected_views else None
    
    return clinical_normalizer, mri_stats, pet_stats
    
    
