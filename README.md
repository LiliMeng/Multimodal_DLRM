# Multimodal_DLRM

To adapt the MovieLens 25M dataset into a multi-modal input for DLRM, we can integrate additional data modalities like movie posters (images) and plot summaries (text) along with the existing tabular data (e.g., user ratings, movie metadata). Below is an example of how you might structure the code to do this.

### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas torch torchvision transformers scikit-learn
```

### Step-by-Step Implementation

#### 1. **Download the MovieLens Dataset and Additional Modalities**

```python
import pandas as pd

# Load the MovieLens 25M dataset
ratings = pd.read_csv('ml-25m/ratings.csv')
movies = pd.read_csv('ml-25m/movies.csv')
links = pd.read_csv('ml-25m/links.csv')

# Example additional data: load movie posters and plot summaries
# Assume you have pre-downloaded the posters and plot summaries
posters = pd.read_csv('posters.csv')  # MovieID -> Image Path
summaries = pd.read_csv('summaries.csv')  # MovieID -> Plot Summary
```

#### 2. **Preprocess the Data**
- Normalize ratings, tokenize text, and load image features.

```python
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
import torch

# Normalize ratings
ratings['rating'] = ratings['rating'] / ratings['rating'].max()

# Load the pre-trained BERT tokenizer and model for text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Example function to get the movie poster and plot summary features
def get_multimodal_features(movie_id):
    # Load and process the image
    image_path = posters.loc[posters['movieId'] == movie_id, 'poster_path'].values[0]
    image_tensor = load_image(image_path)
    
    # Load and process the text
    summary = summaries.loc[summaries['movieId'] == movie_id, 'summary'].values[0]
    inputs = tokenizer(summary, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    with torch.no_grad():
        text_features = bert_model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token embedding
    
    return image_tensor, text_features
```

#### 3. **Modify DLRM to Accept Multi-modal Inputs**

This part assumes you've implemented or are using a DLRM model that accepts additional modalities.

```python
import torch.nn as nn
import torch

class DLRM_Multimodal(nn.Module):
    def __init__(self, input_size, image_size, text_size, output_size=1):
        super(DLRM_Multimodal, self).__init__()
        # DLRM components
        self.mlp_dense = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Multimodal components
        self.mlp_image = nn.Sequential(
            nn.Linear(image_size, 64),
            nn.ReLU(),
        )
        
        self.mlp_text = nn.Sequential(
            nn.Linear(text_size, 64),
            nn.ReLU(),
        )
        
        # Interaction and final prediction layer
        self.interaction = nn.Linear(192, 128)
        self.prediction = nn.Linear(128, output_size)

    def forward(self, dense_features, image_features, text_features):
        # Process dense features
        dense_out = self.mlp_dense(dense_features)
        
        # Process multimodal features
        image_out = self.mlp_image(image_features)
        text_out = self.mlp_text(text_features)
        
        # Concatenate all features
        combined = torch.cat([dense_out, image_out, text_out], dim=1)
        
        # Interaction and prediction
        interaction_out = self.interaction(combined)
        output = self.prediction(interaction_out)
        return output
```

#### 4. **Create a DataLoader for the Multi-modal Data**

```python
from torch.utils.data import Dataset, DataLoader

class MovieLensDataset(Dataset):
    def __init__(self, ratings, posters, summaries):
        self.ratings = ratings
        self.posters = posters
        self.summaries = summaries

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating_data = self.ratings.iloc[idx]
        movie_id = rating_data['movieId']
        
        # Get tabular data (e.g., user and movie IDs)
        user_id = rating_data['userId']
        movie_id = rating_data['movieId']
        rating = torch.tensor(rating_data['rating'], dtype=torch.float32)

        # Get multi-modal features
        image_tensor, text_tensor = get_multimodal_features(movie_id)
        
        # Dense features can be user/movie IDs or other numerical data
        dense_features = torch.tensor([user_id, movie_id], dtype=torch.float32)

        return dense_features, image_tensor.squeeze(), text_tensor.squeeze(), rating

# Instantiate dataset and dataloader
dataset = MovieLensDataset(ratings, posters, summaries)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### 5. **Train the Multi-modal DLRM**

```python
import torch.optim as optim

# Instantiate the model
model = DLRM_Multimodal(input_size=2, image_size=2048, text_size=768)  # Adjust sizes based on feature vectors
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for dense_features, image_features, text_features, rating in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(dense_features, image_features, text_features)
        loss = criterion(output.squeeze(), rating)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
```

### Summary
This code integrates movie posters and plot summaries with the MovieLens 25M dataset, processes them through DLRM, and trains the model on these multi-modal inputs. Depending on your specific needs, you might adjust the architecture and preprocessing steps.

## Multimodal_DLRM with CLIP

Integrating the above multi-modal MovieLens code with a model like CLIP (Contrastive Language-Image Pre-training) involves leveraging CLIP's ability to encode both images and text into a shared latent space. This can be done by using CLIP to process the movie posters (images) and plot summaries (text) and then feeding the resulting embeddings into the DLRM model.

### Prerequisites
Ensure you have the `CLIP` model from OpenAI installed:

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Updated Implementation

#### 1. **Load and Preprocess Data with CLIP**

```python
import torch
import clip
from PIL import Image

# Load the CLIP model and preprocess function
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Preprocess function for images using CLIP
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0).to(device)

# Function to tokenize text using CLIP
def preprocess_text(text):
    return clip.tokenize([text]).to(device)

# Function to get CLIP embeddings
def get_clip_embeddings(movie_id):
    # Load and preprocess the image
    image_path = posters.loc[posters['movieId'] == movie_id, 'poster_path'].values[0]
    image_tensor = preprocess_image(image_path)
    
    # Load and preprocess the text
    summary = summaries.loc[summaries['movieId'] == movie_id, 'summary'].values[0]
    text_tensor = preprocess_text(summary)
    
    # Get the CLIP embeddings
    with torch.no_grad():
        image_features = model.encode_image(image_tensor).squeeze(0)
        text_features = model.encode_text(text_tensor).squeeze(0)
    
    return image_features, text_features
```

#### 2. **Modify the DLRM Model to Use CLIP Embeddings**

Since CLIP embeddings are already in a shared latent space, we can directly use these embeddings in the DLRM model.

```python
class DLRM_CLIP(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(DLRM_CLIP, self).__init__()
        # DLRM components for tabular data
        self.mlp_dense = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Final prediction layer after combining CLIP embeddings with dense features
        self.interaction = nn.Linear(64 + 512 + 512, 128)  # Adjust for CLIP's output size
        self.prediction = nn.Linear(128, output_size)

    def forward(self, dense_features, image_features, text_features):
        # Process dense features
        dense_out = self.mlp_dense(dense_features)
        
        # Concatenate dense features with CLIP embeddings
        combined = torch.cat([dense_out, image_features, text_features], dim=1)
        
        # Interaction and prediction
        interaction_out = self.interaction(combined)
        output = self.prediction(interaction_out)
        return output
```

#### 3. **Update the DataLoader to Use CLIP**

```python
class MovieLensCLIPDataset(Dataset):
    def __init__(self, ratings, posters, summaries):
        self.ratings = ratings
        self.posters = posters
        self.summaries = summaries

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating_data = self.ratings.iloc[idx]
        movie_id = rating_data['movieId']
        
        # Get tabular data (e.g., user and movie IDs)
        user_id = rating_data['userId']
        movie_id = rating_data['movieId']
        rating = torch.tensor(rating_data['rating'], dtype=torch.float32)

        # Get CLIP-based multi-modal features
        image_features, text_features = get_clip_embeddings(movie_id)
        
        # Dense features can be user/movie IDs or other numerical data
        dense_features = torch.tensor([user_id, movie_id], dtype=torch.float32).to(device)

        return dense_features, image_features, text_features, rating

# Instantiate dataset and dataloader
dataset = MovieLensCLIPDataset(ratings, posters, summaries)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

#### 4. **Train the DLRM Model with CLIP**

```python
# Instantiate the model
model = DLRM_CLIP(input_size=2).to(device)  # Adjust for the input size (user_id, movie_id)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for dense_features, image_features, text_features, rating in dataloader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(dense_features, image_features, text_features)
        loss = criterion(output.squeeze(), rating)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')
```

### Summary
In this approach, the CLIP model processes both movie posters and plot summaries, generating embeddings that are then fed into a modified DLRM model along with tabular data. This integration leverages CLIP's ability to encode images and text into a shared latent space, which can improve the performance of recommendation tasks that involve multi-modal data.

## Evalution code
To evaluate the DLRM model with CLIP embeddings on the MovieLens 25M dataset, you can follow these steps to calculate metrics such as Mean Squared Error (MSE) and potentially other recommendation metrics like Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). Here’s how you can structure the evaluation code:

### Evaluation Code

#### 1. **Prepare the Dataset for Evaluation**
You should split your dataset into training and test sets. Here, we assume you have already split the dataset and are focusing on evaluating the model on the test set.

```python
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

# Split the dataset into train and test
train_indices, test_indices = train_test_split(range(len(ratings)), test_size=0.2, random_state=42)

# Create DataLoader for test set
test_dataset = Subset(MovieLensCLIPDataset(ratings, posters, summaries), test_indices)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

#### 2. **Evaluation Function**

This function will compute the predictions on the test set and calculate evaluation metrics.

```python
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for dense_features, image_features, text_features, rating in dataloader:
            # Forward pass
            output = model(dense_features, image_features, text_features).squeeze()
            loss = criterion(output, rating)
            
            total_loss += loss.item()
            
            all_predictions.extend(output.cpu().numpy())
            all_targets.extend(rating.cpu().numpy())
    
    # Calculate Mean Squared Error
    mse = total_loss / len(dataloader)
    
    # Convert lists to numpy arrays for further metric calculations
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    return mse, mae, rmse

# Instantiate loss function (Mean Squared Error)
criterion = nn.MSELoss()

# Evaluate the model on the test set
mse, mae, rmse = evaluate_model(model, test_dataloader, criterion)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"Test RMSE: {rmse}")
```

#### 3. **Additional Evaluation Metrics (Optional)**
You might also want to consider additional metrics depending on your task. For example, if your model is used in a recommendation system, you might evaluate metrics such as Precision, Recall, or Normalized Discounted Cumulative Gain (NDCG). However, for this regression task, the above metrics should suffice.

### Summary of Evaluation Code
- **Test DataLoader**: The test dataset is loaded into a DataLoader without shuffling to ensure consistent evaluation.
- **Evaluation Function**: The function computes the MSE, MAE, and RMSE, which are common metrics for evaluating regression models.
- **Model Evaluation**: Finally, the model is evaluated, and the results are printed out.

This code provides a framework to assess the performance of your DLRM model with CLIP embeddings, giving you insights into how well the model is predicting ratings based on multi-modal inputs.
