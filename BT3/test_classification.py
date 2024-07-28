import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

# 1. Đọc dữ liệu
train_df = pd.read_csv("Train_samsung.csv")
test_df = pd.read_csv("Test_samsung_noclass.csv")

# 2. Xử lý dữ liệu thiếu
features_cols = train_df.columns[:-1] 
imputer = SimpleImputer(strategy='most_frequent')
train_df[features_cols] = imputer.fit_transform(train_df[features_cols])
test_df[features_cols] = imputer.transform(test_df[features_cols])

for df in [train_df, test_df]:
    df['X3'] = df['X3'].apply(lambda x: random.randint(3, 5) if x == '3+' else x)

# 3. Mã hóa dữ liệu phân loại (One-hot encoding)
categorical_cols = ['X1', 'X2', 'X4', 'X5', 'X11']
train_df = pd.get_dummies(train_df, columns=categorical_cols)
test_df = pd.get_dummies(test_df, columns=categorical_cols)

# 4. Chuẩn hóa dữ liệu số
numerical_cols = ['X3', 'X6', 'X7', 'X8', 'X9', 'X10']
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

# 5. Chuyển đổi nhãn thành số
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(train_df['Class'])

# Chuyển đổi kiểu dữ liệu của X_train, X_val, X_test thành float
X = train_df.drop('Class', axis=1).astype(float)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)
print("Chuẩn bị dữ liệu hoàn tất!")

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.PReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Khởi tạo mô hình
input_size = X_train.shape[1]
hidden_size = 128
output_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

# Chọn hàm mất mát và thuật toán tối ưu
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# In thông tin mô hình
print(model)


epochs = 60
sum_f1 = 0
num_splits = 5

# Sử dụng KFold Cross Validation
kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)

# Huấn luyện và đánh giá mô hình
for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    # Chia dữ liệu thành các tập huấn luyện và validation
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Khởi tạo lại mô hình cho mỗi fold
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    # Khởi tạo optimizer
    optimizer = optim.Adam(model.parameters())

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i in tqdm(range(X_train.shape[0]), desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = torch.tensor(X_train.values[i], dtype=torch.float).to(device)
            labels = torch.tensor(y_train[i], dtype=torch.float).unsqueeze(0).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Đánh giá trên tập validation
        model.eval()
        val_loss = 0
        y_pred_val = []
        with torch.no_grad():
            for i in range(X_val.shape[0]):
                inputs = torch.tensor(X_val.iloc[i].values, dtype=torch.float).to(device)
                labels = torch.tensor(y_val[i], dtype=torch.float).unsqueeze(0).to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float().item()
                y_pred_val.append(predicted)
        # In thông tin cho mỗi fold và epoch
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/X_train.shape[0]:.4f}, Val Loss: {val_loss/X_val.shape[0]:.4f}")

    # Tính toán F1-score trên tập validation của fold hiện tại
    f1 = f1_score(y_val, y_pred_val)
    sum_f1 += f1
    print(f"F1 Score (Fold {fold + 1}): {f1:.4f}")

# In F1-score trung bình trên tất cả các fold
average_f1 = sum_f1 / num_splits
print(f"Average F1 Score: {average_f1:.4f}")

torch.save(model.state_dict(), f'model_{average_f1:.4f}_{epochs}.pth')