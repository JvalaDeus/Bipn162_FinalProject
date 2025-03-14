import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, TensorDataset

class SleepStageClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepStageClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SleepStageModel:
    def __init__(self, data_path, input_dim=3, hidden_dim=64, output_dim=7, learning_rate=0.001, batch_size=32, epochs=20):
        self.data_path = data_path
        self.model = SleepStageClassifier(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.epochs = epochs

    def load_data(self):
        data = pd.read_csv(self.data_path)
        X = data[['emg', 'eog', 'eeg']].values
        y = data['stage'].values
        X = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=self.batch_size, shuffle=False)

    def train_model(self):
        self.model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}')

    def evaluate_model(self):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(y_batch.numpy())
                y_pred.extend(predicted.numpy())
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Classification Report:\n", classification_report(y_true, y_pred))

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    classifier = SleepStageModel(data_path='path_to_your_data.csv')
    classifier.run()