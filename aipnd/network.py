from torch import nn, optim
from torchvision import models
import logging
from utils import *
import numpy as np


def get_pretrained_network(arch):    
    if arch == 'densenet':
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features

    elif arch == 'resnet':
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
    else:
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
        logging.info('Model not found')
    return model, in_features

def create_classifier(model, hidden_units, in_features):    
    classifier = nn.Sequential(        
        nn.Linear(in_features,hidden_units),
        nn.Dropout(),    
        nn.ReLU(),        
        nn.Linear(hidden_units,256),
        nn.ReLU(),         
        nn.Linear(256,128),
        nn.ReLU(),        
        nn.Linear(128,102),
        nn.LogSoftmax(dim = 1))  
    classifier.requires_grad=True
    logging.info('Classifier created')
    return classifier

def set_classifier(model,classifier, device, arch):
    if arch == 'densenet':
        model.classifier = classifier
    elif arch == 'resenet':
        model.fc = classifier    
    model.to(device)
    return model

    
def get_optimizer(arch, model, l_rate):
    if arch == 'densenet':
        optimizer = optim.Adam(model.classifier.parameters(), lr = l_rate) 
    elif arch == 'resnet':
        optimizer = optim.Adam(model.fc.parameters(), lr = l_rate)       
    return optimizer


def freeze_layers(model, arch):
    if arch == 'densenet':
        for param in model.parameters():
            param.requires_grad = False
    return model


def validation(model, testloader,criterion,device):
    test_loss = 0
    accuracy = 0
    model.eval()
    
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)       
        
        output = model.forward(images)
        test_loss = test_loss+criterion(output, labels).item()   
        ps = torch.exp(output)            
        predicted_classes = ps.max(dim=1)[1]

                
        correct_predictions = (labels.data == predicted_classes)
        accuracy = accuracy+correct_predictions.type(torch.FloatTensor).mean()
        
    return test_loss, accuracy


def save_checkpoint(checkpoint_path, model, class_from_index, hidden_units,
                    learning_rate, batch_size, testing_batch_size, arch):
    checkpoint = {'class_from_index': class_from_index,
                  'hidden_units': hidden_units,
                  'learning_rate': learning_rate,
                  'batch_size': batch_size,
                  'testing_batch_size': testing_batch_size,
                  'arch': arch,
                  'state_dict': model.state_dict()}
    torch.save(checkpoint, checkpoint_path)
    logging.info('Checkpoint saved')


def train_network(model, dataloaders, epochs, device, optimizer, arch):
    logging.info('Starting training...')
    print_every = 45
    step = 0
    model = model.to(device)
    training_dataloader, validation_dataloader = dataloaders['training'],  dataloaders['validation']
    len_validation_data = len(validation_dataloader)

    if arch == 'densenet':
        criterion = nn.NLLLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        logging.debug('Epoch: {}'.format(epoch))
        running_loss = 0    
        model.train()

        for inputs, y in training_dataloader:
            step = step+1
            logging.debug('Step: {}'.format(step))
            optimizer.zero_grad()
            inputs, y = inputs.to(device), y.to(device)

            y_hat = model.forward(inputs)
            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()

            running_loss = running_loss+loss.item()

            if step % print_every == 0:
                model.eval()

                with torch.no_grad():
                    test_loss, accuracy = validation(model, validation_dataloader, criterion, device)

                print("Epoch: {}/{}... ".format(epoch+1, epochs),
                      "Loss: {:.3f}".format(running_loss/print_every),
                      "Test Loss: {:.3f}".format(test_loss / len_validation_data),
                      "Test Accuracy: {:.3f}".format(accuracy / len_validation_data))

                running_loss = 0
                model.train()    
    return model


def load_checkpoint(checkpoint_path, device):

    checkpoint = torch.load(checkpoint_path)
    
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']


    model, in_features = get_pretrained_network(arch)
    classifier = create_classifier(model, hidden_units, in_features)
    model = set_classifier(model, classifier, device, arch)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    class_from_index = checkpoint['class_from_index']
    logging.info('Checkpoint Loaded')
    return model, class_from_index, arch


def process_image(image):

    img = Image.open(image)
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, std)])
    img = transformations(img)
    np_image = np.array(img)
    return np_image


def predict(image_path, model, device, cat_to_name, class_from_index,  topk=5):

    image = process_image(image_path)
    image = torch.from_numpy(image)
    image = image.to(device)

    model.eval()

    image.unsqueeze_(0)

    output = model.forward(image)

    ps = torch.exp(output)

    probs, classes = ps.topk(topk)
    probs, classes = probs.cpu().detach().numpy(), classes.cpu().detach().numpy()

    categories = [cat_to_name[class_from_index[idx]] for idx in classes[0]]


    true_idx = image_path.split('/')[2]
    true_label = cat_to_name[true_idx]
    print(sum(probs))
    predictions = zip(categories, probs.T)

    print()
    print('##############################')
    print()
    print('  TOP {} probabilities are:'.format(topk))
    print()

    for cat, pred in predictions:
        print('  {}: {}%'.format(cat, round(pred.item(),2)))
    print()
    print('  True label: {}'.format(true_label))
    print()
    print('##############################')
    print()