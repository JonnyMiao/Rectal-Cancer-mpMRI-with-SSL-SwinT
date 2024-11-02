import json
from monai import transforms, data
from monai.data import pad_list_data_collate


def get_loader(args):
    '''
    :param args:
    :return:
    '''
    input_keys = ["images1", "images2", "images3"]
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=input_keys),
            transforms.EnsureChannelFirstd(keys=input_keys),
            transforms.SpatialPadd(keys=input_keys, spatial_size=(128, 128, 16)),
            transforms.Resized(keys=input_keys, spatial_size=(-1, -1, 16),
                               mode=['trilinear', 'trilinear', 'trilinear']),
            transforms.GaussianSmoothd(keys=input_keys, sigma=0.2),
            transforms.ScaleIntensityd(keys=input_keys, minv=-1., maxv=1.),
            transforms.RandFlipd(
                keys=input_keys,
                prob=0.2,
                spatial_axis=0),
            transforms.RandFlipd(
                keys=input_keys,
                prob=0.2,
                spatial_axis=1),
            transforms.RandFlipd(
                keys=input_keys,
                prob=0.2,
                spatial_axis=2),
            transforms.RandRotate90d(
                keys=input_keys,
                prob=0.2,
                max_k=3,
            ),
            transforms.RandScaleIntensityd(
                keys=input_keys,
                factors=0.2,
                prob=0.2
            ),
            transforms.RandShiftIntensityd(
                keys=input_keys,
                offsets=0.2,
                prob=0.2
            ),
            transforms.ToTensord(keys=input_keys),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=input_keys),
            transforms.EnsureChannelFirstd(keys=input_keys),
            transforms.SpatialPadd(keys=input_keys, spatial_size=(128, 128, 16)),
            transforms.Resized(keys=input_keys, spatial_size=(-1, -1, 32),
                               mode=['trilinear', 'trilinear', 'trilinear']),
            transforms.GaussianSmoothd(keys=input_keys, sigma=0.2),
            transforms.ScaleIntensityd(keys=input_keys, minv=-1., maxv=1.),
            transforms.ToTensord(keys=input_keys),
        ]
    )

    if not args.test_mode:
        train_files = json.load(open('./dataset/train_set.json', 'r'))
        val_files = json.load(open('./dataset/val_set.json', 'r'))
        train_ds = data.CacheDataset(
            data=train_files,
            cache_num=24,
            transform=train_transform,
            cache_rate=1.0,
            num_workers=args.workers,
        )

        train_loader = data.DataLoader(train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=args.workers,
                                       persistent_workers=True,
                                       collate_fn=pad_list_data_collate)

        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     persistent_workers=True)

        return [train_loader, val_loader]
    else:
        test_files = json.load(open('./dataset/test_set.json', 'r'))
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=2,
                                      persistent_workers=True)

        return test_loader
