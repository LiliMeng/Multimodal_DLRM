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
