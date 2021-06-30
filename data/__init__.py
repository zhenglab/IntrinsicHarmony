"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import torch.utils.data
from data.base_dataset import BaseDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    # data_loader = CustomDatasetDataLoader(opt)
    # dataset = data_loader.load_data()

    dataset_class = find_dataset_using_name(opt.dataset_mode)
    dataset = dataset_class(opt)
    print("dataset [%s] was created" % type(dataset).__name__)

    # batch_size = int(opt.batch_size / max(1, opt.NUM_GPUS))
    if opt.isTrain==True:
        shuffle = True
        drop_last = True
    elif opt.isTrain==False:
        shuffle = False
        drop_last = False

    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if opt.NUM_GPUS > 1 else None

    # Create a loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=int(opt.num_threads),
        drop_last=drop_last,
        pin_memory=True,
    )
    return dataloader


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)

        batch_size = int(opt.batch_size / max(1, opt.NUM_GPUS))
        if opt.isTrain==True:
            shuffle = True
            drop_last = True
        elif opt.isTrain==False:
            shuffle = False
            drop_last = False

        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset) if opt.NUM_GPUS > 1 else None

        # Create a loader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=(False if self.sampler else shuffle),
            sampler=self.sampler,
            num_workers=int(opt.num_threads),
            drop_last=drop_last,
        )

        # self.dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=opt.batch_size,
        #     shuffle=not opt.serial_batches,
        #     num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    # def __iter__(self):
    #     """Return a batch of data"""
    #     for i, data in enumerate(self.dataloader):
    #         if i * self.opt.batch_size >= self.opt.max_dataset_size:
    #             break
    #         yield data

def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    # sampler = (
    #     loader.batch_sampler.sampler
    #     if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
    #     else loader.sampler
    # )
    sampler = loader.sampler
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)





