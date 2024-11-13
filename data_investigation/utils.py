
import torch
import numpy as np

def make_sequences(df, 
                   historic_sequence_length, 
                   prediction_sequence_length,
                   historic_features,
                   future_features,
                   future_one_hots,
                   historic_one_hots,
                   target_feature,
                   static_features,
                   include_historic_target,
                   return_indices=False,
                   device='cuda' if torch.cuda.is_available() else 'cpu',
                   dtype=torch.float32
                   ):
    """
    Function to make sequences from a pandas dataframe. The function will return a list of dictionaries where each dictionary contains the historic and future data for a sequence. The function will also return a list of indices if return_indices is set to True.

    Input Arguments
    ----------
    df : pandas.DataFrame
        The dataframe to make sequences from.
    historic_sequence_length : int
        The length of the historic sequence.
    prediction_sequence_length : int
        The length of the prediction sequence.
    historic_features : list or str
        A list of the historic features to include in the sequences.
    future_features : list or str
        A list of the future features to include in the sequences.
    target_feature : str
        The target feature to predict.
    static_features : list or str
        A list of static features to include in the sequences.
    include_historic_target : bool
        Whether to include the target feature in the historic features.
    return_indices : bool
        Whether to return the indices of the sequences.

    Returns
    -------
    sequences_x : list
        A list of dictionaries where each dictionary contains the selected features data for a sequence.
    sequences_y : list
        A list of the target values for each sequence.
    indices : list
        A list of indices for each sequence.
   
    """
    df = df.copy()
    # iteration length
    n = len(df) - historic_sequence_length - prediction_sequence_length

    # make sure that historic_features and future_features are lists
    if type(historic_features) != list:
        historic_features = [historic_features]
    if type(future_features) != list:
        future_features = [future_features]
    if type(target_feature) != list:
        target_feature = [target_feature]
    if type(static_features) != list:
        static_features = [static_features]
    if type(historic_one_hots) != list:
        historic_one_hots = [historic_one_hots]
    if type(future_one_hots) != list:
        future_one_hots = [future_one_hots]
    
    one_hots = list(set(historic_one_hots + future_one_hots))
    for one_hot in one_hots:
        if (one_hot != None) and (one_hot != False) and (one_hot not in df.columns):

            # one_hot must be meant to have to do with the index
            df[one_hot] = getattr(df.index, one_hot.lower())

    # static features
    static_features = torch.Tensor(np.array(static_features,dtype=float))

    # collect all historic features
    all_historic_features = historic_features + target_feature if include_historic_target else historic_features
    
    # remove None from all_historic_features
    all_historic_features = list(set([feature for feature in all_historic_features if feature is not None]))
    all_historic_features_incl_one_hots = list(set([feature for feature in all_historic_features + historic_one_hots if feature is not None]))

    # collect all future features
    all_future_features = future_features + target_feature + future_one_hots

    # remove None from all_future_features
    all_future_features = np.unique([feature for feature in all_future_features if feature is not None])
    # print('all future features', all_future_features)

    # make slices of historic and future data
    df_historic = df[all_historic_features_incl_one_hots]
    df_future = df[all_future_features]
    torch_historic = torch.Tensor(df_historic.values).to(dtype=dtype, device=device)
    torch_future = torch.Tensor(df_future.values).to(dtype=dtype, device=device)
    torch_index = df[all_future_features].index


    # get indices of columns of features
    future_features_col_ind = [df_future.columns.get_loc(feature) for feature in future_features if feature is not None]
    # print('future features col ind', future_features_col_ind)
    all_historic_features_col_ind = [df_historic.columns.get_loc(feature) for feature in all_historic_features]
    target_feature_col_ind = df_future.columns.get_loc(target_feature[0])

    future_one_hots_ind = [df_future.columns.get_loc(feature) for feature in future_one_hots if feature is not None]
    future_unique_one_hots = []
    for future_one_hot in future_one_hots:
        if future_one_hot is not None:
            future_unique_one_hots.append(len(df_future[future_one_hot].unique()))

    historic_one_hots_ind = [df_historic.columns.get_loc(feature) for feature in historic_one_hots if feature is not None]
    historic_unique_one_hots = []
    for historic_one_hot in historic_one_hots:
        if historic_one_hot is not None:
            historic_unique_one_hots.append(len(df_historic[historic_one_hot].unique()))
                    

    # number of historic features
    n_historic_features = len(all_historic_features)
    
    # if dict_or_tensor == 'dict':
    return iterate_dict(
        torch_future, 
        torch_historic,
        torch_index,
        historic_sequence_length, 
        prediction_sequence_length, 
        target_feature_col_ind, 
        future_features, 
        future_one_hots,
        future_one_hots_ind,
        future_unique_one_hots,
        historic_one_hots,
        historic_one_hots_ind,
        historic_unique_one_hots,
        static_features, 
        n_historic_features, 
        all_historic_features_col_ind, 
        future_features_col_ind, 
        return_indices,
        n
    )
    

def iterate_dict(
        torch_future, 
        torch_historic, 
        torch_index,
        historic_sequence_length, 
        prediction_sequence_length, 
        target_feature_col_ind, 
        future_features, 
        future_one_hots,
        future_one_hots_ind,
        future_unique_one_hots,
        historic_one_hots,
        historic_one_hots_ind,
        historic_unique_one_hots,
        static_features, 
        n_historic_features, 
        all_historic_features_col_ind, 
        future_features_col_ind, 
        return_indices,
        n
        ):

    
    sequences_x = []
    sequences_y = []
    indices = []
    # skip_historic_count = 0
    # skip_future_count = 0

    # iterate over all sequences
    for i in range(n):

        # make placeholder for x data
        x = {}

        # add future data if not na 
        torch_future_slice = torch_future[i+historic_sequence_length:i+historic_sequence_length+prediction_sequence_length]

        # nr_interpolated_vals = df_future_slice[]

        # check if there are any na values in the slice, if so, skip this slice
        if torch.isnan(torch_future_slice).any().item():
            # skip_future_count += 1
            continue
        
        # add historic data if we have more than one feature
        if n_historic_features > 0:
            torch_historic_slice = torch_historic[i:i+historic_sequence_length]

            # check if there are any na values in the slice, if so, skip this slice
            if torch.isnan(torch_historic_slice).any().item():
                # skip_historic_count += 1
                continue
            
            x['historic'] = torch_historic_slice[:, all_historic_features_col_ind]  #torch.tensor(df_historic_slice[all_historic_features].values).to(torch.float32)

        # add future data to x if not None
        if future_features[0] is not None:
            x['future'] = torch_future_slice[:,future_features_col_ind] #torch.tensor(df_future_slice[future_features].values).to(torch.float32)

        if future_one_hots[0] is not None:
            for j, (ind, one_hot_feature) in enumerate(zip(future_one_hots_ind, future_one_hots)):
                one_hot_tensor = torch_future_slice[:,ind]
                one_hot_tensor = torch.nn.functional.one_hot(one_hot_tensor.to(torch.int64), num_classes=future_unique_one_hots[j]).type(torch.bool)
                x[one_hot_feature+'_future_one_hot'] = one_hot_tensor#.to(torch.float32)
            
        if historic_one_hots[0] is not None:
            for j, (ind, one_hot_feature) in enumerate(zip(historic_one_hots_ind, historic_one_hots)):
                one_hot_tensor = torch_historic_slice[:,ind]
                one_hot_tensor = torch.nn.functional.one_hot(one_hot_tensor.to(torch.int64), num_classes=historic_unique_one_hots[j]).type(torch.bool)
                x[one_hot_feature+'_historic_one_hot'] = one_hot_tensor#.to(torch.float32)

        # add static features to x if not None
        if not torch.isnan(static_features).any().item():
            x['static'] = static_features #torch.tensor(static_features).to(torch.float32)

        # x['target'] = target_feature

        # update sequences
        sequences_x.append(
                x
        )

        sequences_y.append(
            torch_future_slice[:, target_feature_col_ind] #torch.tensor(df_future_slice[target_feature].values).to(torch.float32)
            #torch.tensor(df_future_slice[target_feature].values).to(torch.float32)
        )

        # if return_indices add indices
        if return_indices:

            indices.append(torch_index[i+historic_sequence_length:i+historic_sequence_length+prediction_sequence_length])


    if len(sequences_y) >0:
        sequences_y = torch.stack(sequences_y)
    else:
        sequences_y = torch.tensor([])
    if return_indices:
        return sequences_x, sequences_y, indices
    else:
        return sequences_x, sequences_y
    

