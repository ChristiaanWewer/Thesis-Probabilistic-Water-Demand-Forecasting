import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        
        # configuration
        self.in_features = config['historic_sequence_length']
        self.output_size = config['forecast_sequence_length']
        self.dropout_rate = config['dropout_rate']
        self.hidden_size = config['hidden_size']
        self.data_keys = config['data_keys']
        self.hidden_layers = config['hidden_layers']

        for key in self.data_keys:
            if key == 'future':
                self.in_features += self.output_size
            if key == 'Weekday_future_one_hot':
                self.in_features += 7*self.output_size
            if key == 'Hour_future_one_hot':
                self.in_features += 24*self.output_size
            if key == 'Weekday_historic_one_hot':
                self.in_features += 7*self.in_features
            if key == 'Hour_historic_one_hot':
                self.in_features += 24*self.in_features
            if key == 'static':
                self.in_features += config['static_size']

        self.first_dropout = nn.Dropout(p=self.dropout_rate)

        # layers
        if self.hidden_layers > 0:
            self.hidden_units = nn.Sequential(*[nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size),
                nn.Dropout(p=self.dropout_rate),
                nn.ReLU()) for _ in range(self.hidden_layers)]
            )

        # layers
        self.input_layer = nn.Linear(in_features=self.in_features, out_features=self.hidden_size)
        self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)


    def forward(self, x):

        # get data from dictionary
        batch_size = x[self.data_keys[0]].shape[0]
        
        # concatenate all data keys
        x = torch.cat([x[key].reshape(batch_size, -1) for key in self.data_keys], dim=-1)

        # pass through input layer
        x = self.input_layer(x)

        # apply dropout
        x = self.first_dropout(x)

        # pass through hidden layers
        x = nn.ReLU()(x)

        # print dimensions of hidden units
        # pass through output layer
        if self.hidden_layers > 0:
            x = self.hidden_units(x)

        # pass through output layer
        x = self.output_layer(x)
        
        return x