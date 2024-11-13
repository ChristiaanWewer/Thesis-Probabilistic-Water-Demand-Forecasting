import torch.nn as nn
import torch

class LSTMEncodedStatic(nn.Module):
    def __init__(self, config):
        super(LSTMEncodedStatic, self).__init__()

        # lstm
        self.LSTM_hidden_size = config['LSTM_hidden_size']
        self.LSTM_num_layers = config['LSTM_num_layers']
        self.output_size = config['forecast_sequence_length']
        self.one_hot_output_size = config['one_hot_output_size']

        # static encoder layers
        self.static_size = config['static_size']
        self.sequence_length = config['historic_sequence_length']

        self.LSTM_input_size = 1 + self.one_hot_output_size

        # dropout
        self.dropout_rate = config['dropout_rate']

        # dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        # static encoder
        self.static_encoder = nn.Linear(in_features=self.static_size, out_features=self.one_hot_output_size, bias=False)
        # self.static_encoder == nn.linear(in_features=self.static_size, out_features=self.LSTM_input_size)

        self.sm = nn.Softmax(dim=2)

        # lstm
        self.lstm = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            batch_first=True,
            dropout=self.dropout_rate if self.LSTM_num_layers > 1 else 0,
        )

        # output layer
        self.output_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=self.output_size)

    def forward(self, x):

        # static features
        xs = x['static'] #.to(torch.float32)

        xs = self.static_encoder(xs.repeat(1, self.sequence_length,1))#.#unsqueeze(1).repeat(1,self.sequence_length,1)
        xs = self.sm(xs)
        # print(xs.shape)

        # concatenate historic and static features
        x = torch.cat((x['historic'], xs), dim=-1)
        # print(x.shape)
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