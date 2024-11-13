import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler

class LSTMStatic(nn.Module):
    def __init__(self, config):
        super(LSTMStatic, self).__init__()

        # lstm
        self.LSTM_hidden_size = config['LSTM_hidden_size']
        self.LSTM_num_layers = config['LSTM_num_layers']
        self.output_size = config['forecast_sequence_length']
        # self.one_hot_output_size = config['one_hot_output_size']

        # static encoder layers
        self.static_size = config['static_size']
        self.sequence_length = config['historic_sequence_length']
        self.LSTM_input_size = config['LSTM_input_size']

        # dropout
        self.dropout_rate = config['dropout_rate']

        # dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        # static encoder
        # self.static_encoder = nn.Linear(in_features=self.static_size, out_features=self.one_hot_output_size, bias=False)
        # self.static_encoder == nn.linear(in_features=self.static_size, out_features=self.LSTM_input_size)

        # self.sm = nn.Softmax(dim=1)

        # lstm
        self.lstm = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            batch_first=True,
            dropout=self.dropout_rate
        )

        # output layer
        self.output_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=self.output_size)

    def forward(self, x):

        # static features
        xs = x['static'].squeeze(1)#.to(torch.float32)

        # turn into categorical variable
        xs = (torch.argmax(xs, dim=1).unsqueeze(1).float()).repeat(1, self.sequence_length).unsqueeze(2)

        xs = xs/2.87228132 - 1.5666989

        # concatenate historic and static features
        x = torch.cat((x['historic'], xs), dim=-1)
        # print(x.shape)
        # print(x[0,:,:10])

        # pass through lstm to predict the future
        x, _ = self.lstm(x)   
        
        # get the last hidden state
        x = x[:, -1, :]

        # apply dropout
        x = self.dropout_layer(x)

        # pass through output layer
        x = self.output_layer(x)

        return x