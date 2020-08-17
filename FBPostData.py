import torch.utils.data as data


class FBPostData(data.Dataset):

    def __init__(self, specs, labels):
        self.specs = specs
        self.classes = labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (specs, target) where target is class_index of the target class.
        """
        sample = self.specs[index]
        target = self.classes[index]

        return [sample, target]

    def __len__(self):
        return len(self.specs)
