# import torch
# from torch import nn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = nn.BCEWithLogitsLoss()

class Classifier(nn.Module):

    def __init__(self, input_dim):
        super(Classifier, self).__init__()


# output of network should be a single linear unit
# BCEWithLogits applies sigmoid and therefore we do not add sigmoid to final 
# output layer

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim *4),
            nn.ReLU(),
            nn.Linear(input_dim * 4, input_dim *8),
            nn.ReLU(),
            nn.Linear(input_dim*8, input_dim *4),
            nn.ReLU(),
            nn.Linear(input_dim *4 , input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim //2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)

        )

    def forward(self, x):
        return self.classifier(x)


# Training pipeline
# If x is size B x D then y is required size B x 1

def classifier_train(model, optimizer, epochs, loss_fn, train_dl):
    loss_hist_train = [0] * epochs 
    for epoch in range(epochs):
        for x,y in train_dl:
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits,y)
            loss.backward()
            optimizer.step() 
            loss_hist_train[epoch] += loss.item()

        print('Epoch:', epoch)
        print('Loss:', loss_hist_train[epoch])

    return loss_hist_train