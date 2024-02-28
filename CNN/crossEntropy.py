import numpy as np
import matplotlib.pyplot as plt

def cross_entropy_loss(y_true, y_pred):
   
    # y_pred cant be 0 or 1 because of division by zero.
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # executing the cross-entropy loss.
    loss = -np.sum(y_true * np.log(y_pred))
    
    return loss

y_true = np.array([1, 0])

# generates a range of predicted probabilities 
probs = np.linspace(0.01, 0.99, 100)

# calculates the cross-entropy loss for all predicted probabilities
loss_values = [cross_entropy_loss(y_true, np.array([p, 1 - p])) for p in probs]

# makes plot
plt.figure(figsize=(8, 6))
plt.plot(probs, loss_values, label='Cross-Entropy Loss')
plt.xlabel('Predicted Probability for Class 1')
plt.ylabel('Loss')
plt.title('Cross-Entropy Loss vs. Predicted Probability')
plt.legend()
plt.grid(True)
plt.show()