import torch.nn as nn

class LSTMVanilla(nn.Module):
    def __init__(self, config):
        super(LSTMVanilla, self).__init__()

        # lstm
        self.LSTM_hidden_size = config['LSTM_hidden_size']
        self.LSTM_num_layers = config['LSTM_num_layers']
        self.LSTM_input_size = config['LSTM_input_size']
        self.output_size = config['forecast_sequence_length']

        # dropout
        self.dropout_rate = config['dropout_rate']
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        # lstm
        self.lstm = nn.LSTM(
            input_size=self.LSTM_input_size,
            hidden_size=self.LSTM_hidden_size,
            num_layers=self.LSTM_num_layers,
            batch_first=True
        )

        # output layer
        self.output_layer = nn.Linear(in_features=self.LSTM_hidden_size, out_features=self.output_size)

    def forward(self, x):

        # get data from dictionary
        x = x['historic']

        # pass through lstm to predict the future
        x, _ = self.lstm(x)

        # get the last hidden state
        x = x[:, -1, :]

        # apply dropout
        x = self.dropout_layer(x)

        # pass through output layer
        x = self.output_layer(x)
        
        return x