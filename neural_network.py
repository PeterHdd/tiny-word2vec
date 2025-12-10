import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# ----- Step 1. Tiny fake vocabulary -----
vocab = ["dog", "cat", "park"]
vocab_size = len(vocab)
embedding_dim = 2  # 2D so we can visualize easily

# ----- Step 2. Define model -----
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_dim),
    nn.Linear(embedding_dim, vocab_size)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ----- Step 3. Fake training pairs -----
# (input → target): dog→park, cat→park, dog→cat
pairs = [(0, 2), (1, 2), (0, 1)]  # indexes: dog=0, cat=1, park=2

# ----- Step 4. Print initial weights and biases -----
print("\n=== BEFORE TRAINING ===")
print("Embedding weights:")
print(model[0].weight.data)
print("\nLinear layer weights:")
print(model[1].weight.data)
print("\nLinear layer biases:")
print(model[1].bias.data)

# ----- Step 5. Train + animate -----
plt.ion()
fig, ax = plt.subplots(figsize=(6, 5))

for step in range(50):
    total_loss = 0
    for x, y in pairs:

        x = torch.tensor([x])
        y = torch.tensor([y])

        logits = model(x) # forward pass
        loss = criterion(logits, y) # compute loss

        optimizer.zero_grad() # clear old gradients
        loss.backward() # compute gradients via backprop
        optimizer.step() # apply gradient to update embedding and linear layer weights

        total_loss += loss.item()

    # Update plot every few steps
    if (step + 1) % 2 == 0:
        clear_output(wait=True)
        print(f"Step {step+1} → loss: {total_loss:.4f}")

        embeddings = model[0].weight.data
        x_vals = embeddings[:, 0].numpy()
        y_vals = embeddings[:, 1].numpy()

        ax.clear()
        ax.scatter(x_vals, y_vals, color="blue")
        for i, word in enumerate(vocab):
            ax.text(x_vals[i] + 0.05, y_vals[i] + 0.05, word, fontsize=12)

        ax.set_title(f"Learned Word Embeddings (Step {step+1})")
        ax.set_xlabel("Embedding dimension 1")
        ax.set_ylabel("Embedding dimension 2")
        ax.grid(True)
        plt.pause(0.3)

plt.ioff()
plt.show()

# ----- Step 6. Print final weights and biases -----
print("\n=== AFTER TRAINING ===")
print("Embedding weights:")
print(model[0].weight.data)
print("\nLinear layer weights:")
print(model[1].weight.data)
print("\nLinear layer biases:")
print(model[1].bias.data)
