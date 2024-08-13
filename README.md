# Multimodal_DLRM

## Formulate DLRM and MovieLens Recommendation as a regression problem
When using DLRM (Deep Learning Recommendation Model) on the MovieLens 25M dataset as a multimodal recommendation system, the problem is typically framed as a **regression problem**. The task is to predict the exact rating (a continuous value) that a user will give to a movie, rather than simply classifying whether they will like the movie or not.

### Loss Function for DLRM on MovieLens 25M

MovieLens100K (a smaller version of MovieLens 25M): https://github.com/babu-thomas/movielens-posters 

The most commonly used loss function for this setup is the **Mean Squared Error (MSE)**. MSE is suitable for regression tasks where the goal is to minimize the difference between the predicted ratings and the actual ratings provided by users.

<img width="558" alt="Screenshot 2024-08-13 at 9 55 56 AM" src="https://github.com/user-attachments/assets/5791d267-9936-446f-aafd-7310e7903f86">


### Step-by-Step Explanation of the Recommendation System

#### 1. **Data Preparation**
   - **Input Data**: The MovieLens 25M dataset includes user ratings for movies, as well as metadata such as movie genres, user IDs, and movie IDs. In a multimodal setup, you might also incorporate additional data such as movie posters (images) and plot summaries (text).
   - **Feature Engineering**: 
     - **Sparse Features**: User IDs, movie IDs, and categorical data like genres are treated as sparse features.
     - **Dense Features**: Ratings and possibly other numerical data are treated as dense features.
     - **Multimodal Features**: Images and text are transformed into dense vectors using models like CLIP (for images and text) or ResNet (for images) and BERT (for text).

#### 2. **Embedding Layers**
   - **Sparse Features**: Categorical features are mapped to dense vectors using embedding layers. For example, user IDs and movie IDs are converted to dense vectors that capture their latent characteristics.
   - **Multimodal Embeddings**: Image and text embeddings are generated using pre-trained models like CLIP and then fed into the DLRM model as dense vectors.

#### 3. **Interaction Layer**
   - The interaction between the different feature embeddings (sparse, dense, and multimodal) is computed. In DLRM, this typically involves calculating dot products between pairs of feature vectors to capture their pairwise interactions.

#### 4. **Multi-Layer Perceptron (MLP)**
   - The combined features from the interaction layer are passed through a Multi-Layer Perceptron (MLP). The MLP consists of several fully connected layers that learn complex patterns in the data and help predict the final output.

#### 5. **Output Layer**
   - The output layer generates a predicted rating for each user-movie pair. Since this is a regression problem, the output is a continuous value that represents the predicted rating.

#### 6. **Loss Calculation**
   - The predicted ratings are compared to the actual ratings using the Mean Squared Error (MSE) loss function. The model is trained to minimize this loss by adjusting the weights through backpropagation and optimization techniques (e.g., using the Adam optimizer).

#### 7. **Evaluation**
   - After training, the model's performance is typically evaluated using metrics such as MSE, RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and nDCG@10 (Normalized Discounted Cumulative Gain at rank 10) to assess how well the model ranks the items.

### Classification vs. Regression in Recommendation Systems

- **Regression**: As in the case of MovieLens 25M, if the task involves predicting a continuous value like a rating, it is a regression problem. The model needs to output a score or a real-valued number.
  
- **Classification**: If the task involves predicting whether a user will like or dislike a movie (e.g., predicting a binary outcome like "click" or "not click"), it would be a classification problem. In such cases, the loss function might be binary cross-entropy or categorical cross-entropy.

**In summary**, when using DLRM on the MovieLens 25M dataset as a multimodal recommendation system, the problem is framed as a regression task, and the MSE loss function is used to train the model. The system predicts the exact rating a user is likely to give to a movie, leveraging both user interaction data and multimodal content features.

## DLRM
https://arxiv.org/pdf/1906.00091

The **Deep Learning Recommendation Model (DLRM)**, developed by Meta, is a high-performance neural network architecture designed for large-scale recommendation systems. It was introduced to address the unique challenges of recommendation tasks, which typically involve heterogeneous data types, large input sizes, and the need for real-time inference.

### Key Components and Architecture of DLRM

1. **Input Features**:
   - **Sparse Features**: These are categorical features, like user IDs, item IDs, or other discrete attributes. DLRM uses embedding layers to convert these high-dimensional, sparse categorical features into dense vectors. This helps in reducing the dimensionality and capturing the semantics of the categorical data.
   - **Dense Features**: These are continuous numerical features, such as user age, item price, or interaction counts. DLRM processes these features through fully connected layers.

2. **Embedding Layers**:
   - DLRM uses embedding tables to map sparse categorical features to dense representations. These embeddings are learned during training and are crucial for representing the categorical data efficiently.

3. **Interaction Layer**:
   - The interaction between different features is a critical aspect of DLRM. After transforming the sparse and dense inputs into their respective dense embeddings, DLRM computes interactions between these embeddings. The most common form of interaction in DLRM is the **dot product** between feature embeddings, capturing the pairwise interactions between different features.

4. **Multi-Layer Perceptron (MLP)**:
   - DLRM includes an MLP component that processes the interactions of features. This MLP is responsible for learning non-linear patterns and predicting the final output, such as a user's likelihood to click on an item or give a high rating.

5. **Output**:
   - The final output of DLRM is typically a probability score, representing the likelihood of a certain event (e.g., a user clicking on a recommended item).

### Scalability and Efficiency

- **Parallelism**: DLRM is designed to take advantage of parallelism, especially on GPU clusters, making it suitable for large-scale deployments. The model can handle billions of input features and make predictions in real-time, which is critical for serving recommendations to millions of users.
  
- **Distributed Training**: DLRM supports distributed training across multiple GPUs or machines. This is essential for handling the massive datasets typically involved in recommendation tasks.

- **Memory Efficiency**: One of the challenges in recommendation systems is the large size of embedding tables, especially with billions of items. DLRM employs techniques like mixed precision training and quantization to reduce memory usage while maintaining performance.

### Performance and Applications

DLRM has been extensively used in production at Meta for various recommendation tasks, including personalized ads, content ranking, and social media feed optimization. Its architecture is flexible and can be adapted to different types of recommendation scenarios, from predicting user engagement to ranking search results.

### Open Source and Research

Meta has open-sourced DLRM through their [GitHub repository](https://github.com/facebookresearch/dlrm), allowing researchers and engineers to experiment with and extend the model. The open-source release includes code for training, evaluation, and deployment, making it a valuable resource for developing state-of-the-art recommendation systems.

DLRM has had a significant impact on the field of recommendation systems, offering a scalable and efficient approach to handling the complexities of real-world recommendation tasks. Its introduction has spurred further research into deep learning-based recommendation models and has set a benchmark for performance in the industry.

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

## Additional evaluation with nDCG@10

When using the Mean Squared Error (MSE) as the loss function during training in a recommendation system, you can still use the nDCG (Normalized Discounted Cumulative Gain) as an evaluation metric to measure how well your model ranks the items for users. Here's how you can integrate nDCG as an evaluation metric alongside MSE as the loss function:

### 1. **Training with MSE Loss Function**
   - During the training phase, you minimize the MSE loss. MSE measures the average squared difference between the predicted and actual ratings. This helps the model to predict ratings as accurately as possible.

### 2. **Evaluating with nDCG Metric**
   - **nDCG@k** is a ranking-based evaluation metric that measures the quality of the top-k recommendations. It takes into account both the position of relevant items in the ranking and the actual relevance (e.g., rating) of those items. Unlike MSE, which evaluates prediction accuracy, nDCG evaluates how well the model ranks the items.

#### Steps to Calculate nDCG:

1. **Prediction**:
   - After training the model with MSE, for each user in the test set, predict scores for all possible items (movies).

2. **Ranking**:
   - Rank these items based on the predicted scores. For example, for nDCG@10, you would take the top 10 items with the highest predicted scores for each user.

3. **Relevance Scores**:
   - The true relevance scores are typically derived from the actual ratings in the test set. The relevance can be binary (e.g., relevant or not) or multi-level (e.g., the actual rating value).

4. **Compute DCG@k**:
   - Discounted Cumulative Gain (DCG) at position \( k \) is calculated as:
     \[
     \text{DCG@k} = \sum_{i=1}^{k} \frac{\text{relevance}_i}{\log_2(i + 1)}
     \]
     where \( \text{relevance}_i \) is the relevance score of the item at position \( i \) in the ranked list.

5. **Compute IDCG@k**:
   - Ideal DCG (IDCG) is computed by sorting items by their true relevance scores and then computing the DCG for this ideal ranking.

6. **Compute nDCG@k**:
   - The nDCG@k is calculated as:
     \[
     \text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
     \]
     This value ranges from 0 to 1, where 1 indicates a perfect ranking.

### Example Implementation in Code

Assuming you have the predicted scores and the true ratings, here's how you can calculate nDCG@10 in Python:

```python
from sklearn.metrics import ndcg_score

def calculate_ndcg(y_true, y_pred, k=10):
    """
    Calculate the nDCG score at k.
    
    y_true: list of true relevance scores
    y_pred: list of predicted scores
    k: the cutoff rank
    """
    return ndcg_score([y_true], [y_pred], k=k)

# Example data for one user
true_ratings = [5, 4, 3, 2, 1]  # True ratings for all items
predicted_scores = [0.8, 0.3, 0.7, 0.2, 0.5]  # Model predicted scores

# Calculate nDCG@10
ndcg_value = calculate_ndcg(true_ratings, predicted_scores, k=10)
print(f"nDCG@10: {ndcg_value}")
```

### Summary
- **Training**: Use MSE during training to optimize the prediction accuracy.
- **Evaluation**: Use nDCG@k to evaluate how well the model ranks items in the top-k recommendations.

This approach allows you to optimize for prediction accuracy while still ensuring that the model produces useful and relevant rankings for users, which is critical in recommendation systems.

## Additional nDCG@10 metrics

To evaluate the DLRM model with CLIP embeddings using the nDCG@10 (Normalized Discounted Cumulative Gain at rank 10) metric, you can implement the following code. This metric is widely used in recommendation systems to measure the quality of the ranking produced by the model.

### Prerequisites
Make sure you have the necessary packages installed:

```bash
pip install torch scikit-learn
```

### nDCG@10 Calculation Code

#### 1. **Import Necessary Libraries**

```python
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score

# Assuming that you have the MovieLensCLIPDataset and DLRM_CLIP model from previous code
```

#### 2. **Modify the Dataset to Return All Items for a User**

To compute nDCG@10, we need to predict scores for all possible items for each user and then rank them.

```python
class MovieLensCLIPDatasetAllItems(Dataset):
    def __init__(self, ratings, posters, summaries, all_movie_ids):
        self.ratings = ratings
        self.posters = posters
        self.summaries = summaries
        self.all_movie_ids = all_movie_ids

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        rating_data = self.ratings.iloc[idx]
        user_id = rating_data['userId']
        true_movie_id = rating_data['movieId']
        rating = torch.tensor(rating_data['rating'], dtype=torch.float32)

        # Get dense features
        dense_features = torch.tensor([user_id], dtype=torch.float32).to(device)

        # Predict scores for all items
        all_predictions = []
        for movie_id in self.all_movie_ids:
            image_features, text_features = get_clip_embeddings(movie_id)
            image_features, text_features = image_features.to(device), text_features.to(device)
            score = model(dense_features, image_features, text_features).item()
            all_predictions.append(score)
        
        return all_predictions, true_movie_id, rating.item()
```

#### 3. **Evaluate nDCG@10**

```python
def evaluate_ndcg_at_10(model, dataloader):
    model.eval()
    ndcg_scores = []

    with torch.no_grad():
        for all_predictions, true_movie_id, true_rating in dataloader:
            # Convert predictions and ground truth to suitable format
            all_predictions = np.array(all_predictions)
            true_relevance = np.zeros_like(all_predictions)
            true_relevance[true_movie_id] = true_rating
            
            # Calculate nDCG@10
            ndcg = ndcg_score([true_relevance], [all_predictions], k=10)
            ndcg_scores.append(ndcg)
    
    # Average nDCG@10 over all users
    avg_ndcg = np.mean(ndcg_scores)
    return avg_ndcg

# Assuming you have already split your dataset into train and test indices
all_movie_ids = ratings['movieId'].unique()
test_dataset = MovieLensCLIPDatasetAllItems(ratings.iloc[test_indices], posters, summaries, all_movie_ids)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluate the model
avg_ndcg = evaluate_ndcg_at_10(model, test_dataloader)
print(f"Average nDCG@10: {avg_ndcg}")
```

### Explanation

1. **Dataset Modification**: The `MovieLensCLIPDatasetAllItems` class has been modified to return the predicted scores for all items (movies) for a given user. This is essential for calculating nDCG, as it requires ranking all possible items for a user.

2. **nDCG@10 Calculation**: The `evaluate_ndcg_at_10` function computes the nDCG@10 for each user and averages it across all users in the test set. This provides a measure of how well the model ranks relevant items (i.e., those that the user has rated highly).

3. **Average nDCG@10**: The average nDCG@10 across all users is printed as the final evaluation metric.

This code provides a robust method to evaluate the performance of your multi-modal DLRM-CLIP model using the nDCG@10 metric, which is critical for understanding the quality of recommendations generated by the model.

## Baselines:
The state-of-the-art (SOTA) models for the MovieLens 25M dataset primarily focus on recommendation tasks, particularly link prediction and collaborative filtering. As of the latest results:

1. **PEAGAT (Personalized Embedding Attention Graph Autoencoder Transformer)** is currently the SOTA model for link prediction tasks on the MovieLens 25M dataset. This model achieves the best performance by integrating graph-based embeddings with attention mechanisms and transformers, effectively capturing user-item interactions and complex relationships in the data.

2. **LightGT (Light Graph Transformer)** is another leading model that has shown excellent results for multi-media recommendations on the same dataset. It combines graph neural networks (GNNs) and transformers to handle graph-structured data, which is particularly effective for tasks involving collaborative filtering and recommendations.

These models are part of the broader trend in recommendation systems that leverage advanced architectures like transformers and graph neural networks to capture more intricate patterns in user-item interactions. 

The performance of these models is typically evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE), among others, depending on the specific task and benchmark.

You can explore more detailed results and model comparisons on platforms like Papers With Code, which tracks SOTA models across various datasets, including MovieLens 25M (https://paperswithcode.com/sota/link-prediction-on-movielens-25m).

## Dataset details:
The MovieLens 25M dataset is one of the largest and most popular datasets used in research related to recommendation systems, particularly for collaborative filtering techniques. Released by GroupLens, a research lab at the University of Minnesota, the dataset provides an extensive collection of user ratings for movies, along with additional metadata.

### Key Features of the MovieLens 25M Dataset

1. **Size and Scope**:
   - The dataset contains **25 million ratings** on a 5-star scale, spanning from **0.5 to 5 stars**, with increments of 0.5 stars.
   - It includes ratings provided by **162,541 unique users** across **62,423 movies**.
   - The ratings were collected between **January 9, 1995**, and **November 21, 2019**.

2. **Data Composition**:
   - **Ratings Data**: The core of the dataset is the ratings data, where each record represents a rating given by a user to a movie. This is stored in a CSV file with columns for user ID, movie ID, rating, and timestamp (indicating when the rating was given).
   - **Movies Data**: Each movie is associated with metadata, including title, genres, and possibly a release year. This information is useful for content-based filtering or hybrid recommendation approaches.
   - **Tags and Tag Applications**: Users can also tag movies with specific labels, which can be used for analyzing user preferences or enhancing recommendation models with contextual information.
   - **Links Data**: Provides connections between the MovieLens IDs and corresponding IDs from IMDb and TMDb (The Movie Database), allowing for enrichment of the dataset with additional external data such as cast, crew, plot summaries, and posters.

3. **Genres**:
   - The movies are classified into one or more of 20 different genres, such as Action, Comedy, Drama, and Sci-Fi. These genres are useful for both collaborative filtering and content-based recommendations.

4. **User Data**:
   - The dataset does not contain explicit user profiles or demographic data but focuses on the behavioral aspect—primarily the ratings and tags provided by users. The absence of demographic data helps maintain user privacy.

5. **Applications**:
   - **Collaborative Filtering**: The dataset is widely used to develop and test collaborative filtering algorithms, which predict a user’s preference based on the preferences of similar users.
   - **Content-Based Filtering**: The movie metadata, including genres and tags, supports content-based recommendation methods, where movies similar to what the user liked before are recommended.
   - **Hybrid Models**: Combining collaborative filtering with content-based methods, the MovieLens 25M dataset is also used for developing hybrid recommendation systems that leverage both user interaction and movie content.

6. **Challenges and Research**:
   - The dataset poses several challenges, such as sparsity (many movies have only a few ratings), scalability (due to the large number of users and movies), and the cold-start problem (new users or movies with few ratings). These challenges make it a valuable resource for benchmarking the performance of new recommendation algorithms.

7. **Variants**:
   - **MovieLens 1M, 10M, and 20M**: Smaller versions of the dataset are also available, such as MovieLens 1M, 10M, and 20M, which are subsets of the full dataset and are often used in studies where computational resources are limited.

### Research Use and Impact
The MovieLens 25M dataset has been cited in numerous academic papers and has become a standard benchmark in the field of recommendation systems. Its extensive use in the development and testing of recommendation algorithms, especially those based on collaborative filtering, has helped drive advancements in this field. Moreover, the dataset's versatility allows researchers to explore various aspects of recommendation systems, including implicit feedback, temporal dynamics, and explainability.

### Data Access
The MovieLens 25M dataset is freely available for academic and non-commercial use and can be downloaded from the GroupLens website [here](https://grouplens.org/datasets/movielens/25m/).

This dataset continues to be a critical tool for researchers and developers aiming to improve the quality and personalization of recommendation systems.
