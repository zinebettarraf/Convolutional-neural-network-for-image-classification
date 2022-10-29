from torch import nn
from torchvision.models import resnet18


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


class SimpleDetector(nn.Module):
    """ VGG11 inspired feature extraction layers """
    def __init__(self, nb_classes):
        """ initialize the network """
        super().__init__()
        # TODO: play with simplifications of this network
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Flatten()
        )
        self.features.apply(init_weights)

        # create classifier path for class label prediction
        # TODO: play with dimensions of this network and compare
        self.classifier = nn.Sequential(
            # dimension = 64 [nb features per map pixel] x 3x3 [nb_map_pixels]
            # 3 = ImageNet_image_res/(maxpool_stride^#maxpool_layers) = 224/4^3
            nn.Linear(64 * 3 * 3, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, nb_classes)
        )
        self.classifier.apply(init_weights)

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # get features from input then run them through the classifier
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)

# TODO: create a new class based on SimpleDetector to create a deeper model


# TODO: once played with a deeper model, play with this
class ResnetObjectDetector(nn.Module):
    """ Resnet18 based feature extraction layers """
    def __init__(self, nb_classes):
        super().__init__()
        # copy resnet up to the last conv layer prior to fc layers, and flatten
        # TODO: add pretrained=True to get pretrained coefficients: what effect?
        features = list(resnet18(pretrained=True).children())[:9]
        self.features = nn.Sequential(*features, nn.Flatten())

        # TODO: first freeze these layers, then comment this loop to
        #  include them in the training
        # freeze all ResNet18 layers during the training process
        for param in self.features.parameters():
            param.requires_grad = False

        # create classifier path for class label prediction
        # TODO: play with dimensions below and see how it compares
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, nb_classes)
        )

        # create regressor path for bounding box coordinates prediction
        # TODO: take inspiration from above without dropouts

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from two different branches of the network
        x = self.features(x)
        # TODO: compute and add the bounding box regressor term
        return self.classifier(x)
