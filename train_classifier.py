import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Find the maximum length of data samples
max_length = max(len(sample) for sample in data)

# Pad all samples to the same length
padded_data = np.array([
    np.pad(sample, (0, max_length - len(sample)), mode='constant') for sample in data
])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    padded_data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and calculate accuracy
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
