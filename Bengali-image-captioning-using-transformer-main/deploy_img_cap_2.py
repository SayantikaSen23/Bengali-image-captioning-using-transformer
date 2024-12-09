#deploy_img_cap_2.py

import torch
import pickle
import torchvision
from PIL import Image
import random
import math
import matplotlib.pyplot as plt
from model import ImageCaptionModel, PositionalEncoding
from mtranslate import translate

max_seq_len = 33
vocab_size = 8360
k=3



# Load the model and vocab (only once)
def load_model_and_vocab():
    global model, index_to_word, word_to_index, resnet18
    
    # Load the pre-trained model
    model = torch.load('./BestModel', map_location=torch.device('cpu'))

    #Load pretrained resnet18 model
    resnet18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    resnet18.eval()

    # Load vocab
    with open('./vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    index_to_word, word_to_index = vocab['index_to_word'], vocab['word_to_index']
    print(f"Vocabulary loaded. Vocab size: {len(index_to_word)}")

# Call this function at the top level to load everything once
load_model_and_vocab()



"""**Extract features from image using resnet18**"""
def extract_image_features(image, model):
    """ Extract features from the image using ResNet18 """
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model.conv1(image_tensor)  # First conv layer
        features = model.bn1(features)  # Batch norm
        features = model.relu(features)  # ReLU
        features = model.maxpool(features)  # Max pooling
        features = model.layer1(features)  # Layer 1
        features = model.layer2(features)  # Layer 2
        features = model.layer3(features)  # Layer 3
        features = model.layer4(features)  # Layer 4
        features = model.avgpool(features)  # Average Pooling
        features = features.view(features.size(0), -1)  # Flatten to a vector
    return features


# Generate caption for an image outside the dataset
def generate_caption(K, img_path, index_to_word, word_to_index):
    # Load and display the image
    image = Image.open(img_path).convert("RGB")
    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()


    img_embed = extract_image_features(image, resnet18)

    # img_embed is now a flattened vector of shape (batch_size, 512)
    img_embed = img_embed.unsqueeze(1)  # Add sequence length dimension: (batch_size, seq_len, num_features)

    # Prepare input sequence for caption generation
    input_seq = [word_to_index.get('<start>', 0)] * max_seq_len
    input_seq = torch.tensor(input_seq).unsqueeze(0)

    predicted_sentence = []
    model.eval()
    with torch.no_grad():
        for eval_iter in range(0, max_seq_len):
            # Forward pass through the trained captioning model
            output, padding_mask = model.forward(img_embed, input_seq)
            output = output[eval_iter, 0, :]

            # Perform top-K sampling
            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()
            next_word_index = random.choices(indices, values, k=1)[0]

            next_word = index_to_word.get(next_word_index, '<unk>')  # Use '<unk>' for unknown words

            # Update input sequence with the predicted word
            input_seq[:, eval_iter + 1] = next_word_index

            # Stop if the model predicts the end token
            if next_word == '<end>':
                break

            predicted_sentence.append(next_word)

    # Print the predicted caption
    # print("\nPredicted Caption:")
    caption = " ".join(predicted_sentence + ['.'])
    return caption

# Function to translate the generated caption to Bengali using mtranslate
def translate_to_beng(caption):

    # Translate the caption to Bengali using mtranslate
    translated_caption = translate(caption, 'bn', 'en')

    # Return the translated caption
    return translated_caption

def caption_this_image(img_path):
    cap = generate_caption(3, img_path, index_to_word, word_to_index)
    cap = translate_to_beng(cap)
    return cap

