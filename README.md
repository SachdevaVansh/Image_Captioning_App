# IMAGE CAPTIONING PROJECT USING CNN AND LSTM AND DEPLOYMENT USING STREAMLIT

## 🖼️ Image Captioning

Image Captioning is the task of generating a textual description of an image. It lies at the intersection of Computer Vision and Natural Language Processing (NLP). The goal is to interpret visual content and describe it in human language.

### 🔁 How It Works: 
### The Encoder-Decoder Framework 
- Most modern image captioning systems follow an encoder-decoder architecture:
- The encoder processes the image and converts it into a compact vector representation (embeddings).
- The decoder takes this vector and generates a descriptive sequence of words.

### 🧠 Architecture: CNN + RNN (LSTM)
To build a captioning system, we typically combine two deep learning models:

**CNN (Convolutional Neural Network)** 
- Extracts features from the image.
- Produces vector embeddings, whose size depends on the pretrained CNN used (e.g., VGG16, ResNet).

**LSTM (Long Short-Term Memory)**
- Handles the language generation process.
- The image embedding is concatenated with word embeddings and passed to the LSTM.
- It predicts the next word in the caption, step by step.

<p align="center">
  <img src="https://github.com/user-attachments/assets/f938db5e-8149-4c0e-894e-4a407758ee94" alt="image" />
</p>

## 📂 Dataset: Flickr_8K for Image Captioning
For this project, I’m using the Flickr_8K dataset to train the image caption generator. While larger datasets like Flickr_30K and MSCOCO are available, they can take days or even weeks to train effectively. The smaller size of Flickr_8K (~1GB) makes it ideal for faster prototyping and experimentation without requiring extensive computing resources.

🔗 Special thanks to Jason Brownlee for providing a direct download link to the dataset.

## 📁 Dataset Structure

Flicker8k_Dataset/
- Contains 8,000 images of everyday scenes in JPEG format.

Flickr_8k_text/
- Contains important metadata and annotations.

*The key file here is Flickr8k.token.txt, which holds the image-to-caption mappings.*

*Each line contains the image filename and one of its five associated captions, separated by a tab or newline (\n).*

*This dataset is essential for training the model to associate visual features with natural language descriptions.*

### The dataset looks like this :
![Dataset](https://github.com/user-attachments/assets/bd0af1c5-ebb2-4bba-8d3d-2c80761f00f2)

### Visualisation of the Images and thier corresponding captions in the dataset:

![image](https://github.com/user-attachments/assets/4b5d39c4-17ad-46c9-974e-72eab54619a0)

# <u>Step-1 : Caption Text Preprocessing -</u>

This is the procedure that I followed for the Text Preprocessing:

- Converting the sentences into lowercase
- Removing special characters and numbers present in the text
- Removing extra spaces
- Removing single characters
- Adding  a starting and an ending tag to the sentences to indicate the beginning and the ending of a sentence.

The preprocessed text looks like this:

![image](https://github.com/user-attachments/assets/92b7469a-dd5f-4c8c-bb1f-ff0c28bbc370)

# Step-2 : Tokenization and Encoded Representation -

## 🧾 Text Tokenization & Embedding
Before feeding text into the model, each caption (sentence) undergoes the following steps:

### Tokenization ✂️

- Each caption is split into individual words (tokens).

- Example: "A man riding a horse" → ["A", "man", "riding", "a", "horse"]

### One-Hot Encoding 🔢

- Each token is converted into a one-hot vector, where the vector length equals the size of the vocabulary.

- This creates a sparse binary representation for each word.

### Word Embeddings 📦

- These one-hot vectors are passed through an embedding layer, which maps them into dense, lower-dimensional vectors.

- This step captures semantic meaning and contextual similarity between words (e.g., “horse” and “animal” might have similar embeddings).

- This embedding output is then passed to the LSTM model during training to help it learn meaningful patterns in the text data.

*For Example -*

![image](https://github.com/user-attachments/assets/6a63c229-97e0-4856-b9f0-2cf182a7d5fc)

The input sentences are thus then tokenized and the output lokks like - 

![image](https://github.com/user-attachments/assets/50797a4c-2bb0-4eae-8c62-2dcf897676ef)


## 🖼️ Image Feature Extraction
To extract meaningful visual features from the images, I used the DenseNet-201 architecture, a powerful and efficient pretrained convolutional neural network.

> 🔍 Why DenseNet-201?
- It’s known for dense connectivity, where each layer receives input from all previous layers, leading to better feature propagation and strong gradient flow.

- It comes pretrained on ImageNet, providing strong generalization capabilities for transfer learning tasks like image captioning.

> ⚙️ Feature Extraction Setup
- I removed the final classification layers and retained the Global Average Pooling (GAP) layer as the output.

- As a result, each image is transformed into a 1920-dimensional feature vector (embedding).

> 🛠️ Flexibility
- Although DenseNet-201 is used here, other pretrained architectures like VGG16, ResNet50, or InceptionV3 can also be used for feature extraction, depending on performance and speed requirements.

- These 1920-dimensional embeddings are then combined with the textual embeddings during training to generate image captions.

![image](https://github.com/user-attachments/assets/72de0771-ed94-4354-a374-aa36c664a388)


![Workflow Diagram](https://github.com/user-attachments/assets/a00964fb-d540-4a11-89b3-ec366e3cdf2a)

## 🔄 Data Generation
Training an image captioning model is a resource-intensive task. Since loading the entire dataset into memory isn't feasible, I used a data generator to feed data to the model in batches, optimizing memory usage.

> ⚙️ Why Use a Data Generator ?
- Neural networks, especially models combining CNNs and LSTMs, require significant memory.

- A generator yields data batch-by-batch, keeping RAM usage low and training efficient.

> 📥 Inputs for Each Batch
- Image Embeddings: Precomputed 1920-dimensional vectors from DenseNet201 for each image.

- Caption Embeddings: The corresponding caption is tokenized and converted into a sequence of word embeddings.

> 🧠 During Training
- *The model receives:*

- The image embedding

- A partial sequence of the caption (e.g., "A man riding") as input

- The next word in the sequence (e.g., "a") as the target

- This process is repeated word-by-word to help the model learn the structure of the caption.

> 🔍 During Inference
- Captions are generated one word at a time.

- The model uses the previously generated words and the image embedding to predict the next word until the end token is reached.

> 🧠 Modelling
- In the caption generation model, I combined image features with text inputs to produce meaningful descriptions using an LSTM-based architecture.

## 🖼️ + 📝 Input Fusion
- The image embedding (a 1920-dimensional vector from DenseNet201) is concatenated with the start-of-sequence token (<start> or startseq).
- This combined input acts as the initial input to the LSTM network.

## 🔁 Caption Generation
- After receiving the initial input, the LSTM begins generating one word at a time.

- At each time step, the previously generated words are fed back into the model to predict the next word.

- This continues until the model outputs the end-of-sequence token (<end> or endseq), forming a complete sentence.

## 📌 Summary:

- Input: [Image Features] + [startseq]

- Output: A sentence like "A man riding a horse through a field.", generated word-by-word.

This approach enables the model to understand both visual context and linguistic flow, producing accurate and coherent captions.

![Model Architecture](https://github.com/user-attachments/assets/d950675b-6ff9-4a18-8a35-b55d3d7a6ba8)

## Model Losses after the Training :

![image](https://github.com/user-attachments/assets/a140a53e-5dc6-478b-8c7c-fa06d18a7a2f)

## 📈 Learning Curve Analysis
- The training results indicate that the model has overfit the Flickr8k dataset. This is evident from the divergence between training and validation performance — a common issue when working with limited data.

>⚠️ Possible Cause:
- Flickr8k contains only ~8,000 images, which may not be sufficient for the model to generalize well.

## 🛠️ How to Address Overfitting:
**Train on a Larger Dataset 📊**

Upgrade to Flickr30k or MS COCO, which offer richer and more diverse image-caption pairs.

More data helps the model learn broader patterns and reduces the risk of memorizing training examples.

**Use Attention Mechanisms 🎯**

Integrate an attention layer to allow the model to focus on specific regions of the image when generating each word.

Attention improves the model's ability to relate visual features to semantic elements in the caption.

**Use Different Pretrained CNNs 🧠**

Switching to a different feature extractor can improve generalization:

VGG16 / VGG19: Simpler architecture, fewer parameters — less prone to overfitting on small datasets.

InceptionV3: Extracts multi-scale features and uses fewer parameters compared to ResNet or DenseNet.

MobileNet: Lightweight and efficient — good for limited data or resource-constrained environments.

This helps the CNN generalize better to unseen visuals.
👉 These networks may produce smaller embedding sizes than DenseNet201 (e.g., 512 or 2048), which can reduce model complexity and help generalize better.

**Regularization 🔒**
Dropout Layers in the LSTM model (commonly between 0.3–0.5) help prevent co-adaptation of neurons.

Apply L2 regularization (weight decay) on dense or LSTM layers.

**Data Augmentation 🖼️**
Increase dataset variability without collecting new data:

Apply transformations like rotation, flipping, zooming, cropping, brightness shift on input images.

**Reduce Model Complexity ⚙️**
Use a smaller LSTM hidden size or fewer layers to reduce overfitting on small datasets.

Consider using GRU instead of LSTM — fewer parameters, sometimes similar performance.

**Use Pretrained Word Embeddings 🧾**
Instead of learning embeddings from scratch:

Use GloVe or Word2Vec for caption tokens.

They provide rich semantic information and reduce the need for large training data.

## Model saving:
- Saving the Model, Feature Extractor and the Tokenizer for loading the trained model and predicting on the test images to check the Model performance.

## Inferencing 

### 🛠️ Caption Generation Utility Functions ( Greedy Search Method) 
These utility functions are responsible for generating captions during inference time — i.e., when a new image is passed to the trained model.

> 🔄 How It Works:
- Image Embedding Input 🖼️
The process begins by passing the image embedding (extracted via CNN) to the model, along with a start token (e.g., "startseq").

- Word-by-Word Prediction 🧠

At each step, the model predicts the next word in the sequence based on:

The image embedding

The sequence of words generated so far

The newly predicted word is then appended to the input sequence.

- Loop Until End Token 🔚

This continues until the model outputs an end token (e.g., "endseq") or reaches a maximum caption length.

### 📦 Utility Functions Typically Include:
generate_caption() — Generates the full caption for a given image embedding.

word_for_id() — Maps the predicted index back to its corresponding word.

preprocess_text() — Cleans and tokenizes input captions (optional).

beam_search_decoder() — (Optional) Improves caption quality using beam search instead of greedy decoding.

### <i>🔄 Alternatives to Greedy Search:
If you want more diverse or accurate captions, you can explore:

Beam Search: Keeps the top k most likely sequences at each step (better, but slower).

Top-k Sampling / Top-p Sampling: Adds randomness to generate varied captions (used in more creative tasks).</i>

## 📏 BLEU Score for Evaluation
- To evaluate the performance of my image captioning model during inference, I used the BLEU (Bilingual Evaluation Understudy) score — a popular metric for assessing the quality of generated text against human-written references.

- During testing, I compared the captions generated by the model with the reference captions provided in the dataset. The BLEU score helped me quantify how closely the model’s predictions matched the ground truth by analyzing n-gram overlaps and applying a brevity penalty to avoid overly short captions.

- I primarily computed BLEU-1 through BLEU-2 scores to capture both the individual word accuracy and short phrase accuracy, offering a more comprehensive understanding of the model’s language generation capabilities.

## Model Predictions on a random Test Image:

![image](https://github.com/user-attachments/assets/fd8d1a93-51b3-46cf-a9e5-b7abf541fcb6)

##  Model Deployment using Streamlit :
I downloaded the model.keras, tokenizer.pkl and feature_extractor.keras files from the Kaggle GPU notebook and created a Python Project using VS Code as IDE.
Created a virtual environment using Conda and importing all the necessary libraries and dependencies for running this project.

- Created a main.py file with the Streamlit app functionalities and created a simple UI for prediction of the Image Captions when an user uploads an image using the App.

You may see the demo of the live Project below:
![demo](https://github.com/user-attachments/assets/0033b4c1-e993-435a-83db-7f5f3da650dc)






