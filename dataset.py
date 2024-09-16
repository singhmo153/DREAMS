from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.preprocessing import normalize
from sklearn.preprocessing import RobustScaler
import config

def read_npy(path):
    return np.load(path, allow_pickle=True)

def cleanXy(Xy):
    return [x for x in Xy if type(x[1])!=tuple and len(x[1].shape) != 1]

class EngageWanderDataset(Dataset):
    def __init__(self, npy_path, fold_path, mode='mtl', scaler=None, train=True):
        """
        Args:
            npy_path (Array[object]) - Path to npy file containing all data points
            fold_path (List[str]) - train/test fold path containing list of paths
            mode (str) - mtl/engage/wander
            scale (bool) - feature scaling
        """
        assert mode in ['engage', 'wander', 'mtl'], "Invalid Mode"
        assert scaler is not None, "Please provide a feature scaler"
        
        self.mode = mode
        features_label_map = {}
        self.scaler = None
        self.x = []
        self.y = []
        
        stats = {
            
        }
        
        data = read_npy(npy_path)
        stats['total'] = len(data)
        data = cleanXy(data)
        stats['total_cleaned'] = len(data)
        
        for xy in data:
            # changing wander 'skip' label to 'not-attentive/mind-wandered' label
            if xy[3] == 2:
                xy[3] = 1
            fname = xy[0]
            feature = xy[1]
            labels = (xy[3], xy[2])
            
            features_label_map[fname] = (feature, labels)
        
        stats['total_filtered'] = len(features_label_map)
        stats['errs'] = 0
        with open(fold_path) as f:
            fold_data = [i.strip('\n') for i in f.readlines()]
            for ele in fold_data:
                try:
                    xy = features_label_map[ele]                    
                    self.x.append(xy[0])
                    self.y.append(xy[1])

                except KeyError as e:
                    stats['errs'] += 1

        # feature scaling
        X = np.array(self.x)
        N, timesteps, features = X.shape
        X_reshaped = X.reshape(-1, features)
        if train:    
            scaler.fit(X_reshaped)

        scaler.transform(X_reshaped)
        self.x = np.reshape(X_reshaped, (N, timesteps, features))
        self.scaler = scaler
        self.y = np.array(self.y)

        train_y_w = self.y[:, 0]
        train_y_e = self.y[:, 1]

        self.class_weight_w = [
            (1 / self.y[train_y_w==k].shape[0]) * (len(self.y) / 2) for k in np.unique(train_y_w)
        ]
        self.class_weight_e = [
            (1 / self.y[train_y_e==k].shape[0]) * (len(self.y) / 4) for k in np.unique(train_y_e)
        ]

    def __getitem__(self, idx):
        if self.mode == 'mtl':
            return {
                'features':self.x[idx],
                'labels': self.y[idx]
            }
        elif self.mode == 'engage':
            return {
                'features1':self.x[idx],
                'labels': self.y[idx][1]
            }
        elif self.mode == 'wander':
            return {
                'features1':self.x[idx],
                'labels': self.y[idx][0]
            }
        
    def __len__(self):
        return len(self.x)
    
    def get_scaler(self):
        assert self.scaler is not None, "Scaler not available"
        return self.scaler
    
    def class_weights(self):
        return torch.tensor(self.class_weight_w), torch.tensor(self.class_weight_e)
    
def get_dataloaders(path, train_fold, test_fold, mode='mtl', num_workers=8, batch_size=32):
    """
    Returns train and test dataloaders
    """
    scaler = RobustScaler() 
    train_dataset = EngageWanderDataset(path, fold_path=train_fold, scaler=scaler, train=True)
    scaler_train = train_dataset.get_scaler()
    test_dataset = EngageWanderDataset(path, fold_path=test_fold, scaler=scaler_train, train=False)
    
    print ("Train dataset: ", len(train_dataset))
    print ("Test dataset: ", len(test_dataset))
    
    _, _, feature_shape = train_dataset.x.shape
    class_weights = train_dataset.class_weights()

    train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return train, test, feature_shape, class_weights
