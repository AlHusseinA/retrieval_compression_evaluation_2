
# write a custom dataloader class for the cub200 dataset
from datasets.cub200 import Cub2011
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 



class DataLoaderCUB200:
    def __init__(self, data_root, batch_size=32, num_workers=10):
        """
        Initializes the DataLoaderCUB200 class with specified batch size, number of workers, and data root.
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = v2.Compose([
            # v2.Lambda(pad),
            v2.Resize(256),
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.TrivialAugmentWide(),            
            # v2.CutMix(cutmix_alpha=1.0, num_classes=200),
            v2.ToTensor(), 
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
   

        self.test_transform = transforms.Compose([
            # v2.Lambda(pad),
            v2.Resize((224, 224)),  # Resize images to the size expected by ResNet50
            v2.CenterCrop(256),  # Center crop the image
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = Cub2011(self.data_root, train=True, download=False, transform=self.train_transform)
        self.test_dataset = Cub2011(self.data_root, train=False, download=False, transform=self.test_transform)

    # In your DataLoaderCUB200 class
    def get_number_of_classes(self):
        # Assuming that the number of classes is the same in both train and test datasets
        # Get the label to name mapping
        self.label_to_name_train = self.train_dataset.get_class_names()
        self.label_to_name_test = self.test_dataset.get_class_names()

        return self.train_dataset.get_number_of_classes(), self.label_to_name_train, self.label_to_name_test
    def get_unique_ids(self):
        train_image_ids = self.train_dataset.get_train_image_ids()
        test_image_ids = self.test_dataset.get_test_image_ids()
        return train_image_ids, test_image_ids


    def get_dataloaders(self):
        """
        Creates and returns data loaders for the CUB-200-2011 dataset (both training and testing sets).
        
        Returns:
        tuple: A tuple containing the training and testing data loaders.
        """
        trainloader = DataLoader(dataset=self.train_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers, 
                                shuffle=True)

        testloader = DataLoader(dataset=self.test_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers, 
                                shuffle=False)

        return trainloader, testloader
