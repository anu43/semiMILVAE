from dataloaders.warwick_bags_loader import ColonCancerBagsCross


def load_warwick(train_fold, val_fold, test_fold, padding=True, base_att=False):
    # Prepare the training, the validation, and the test sets
    train_set, val_set, test_set = load_warwick_cross(
        train_fold, val_fold, test_fold, padding=padding, base_att=base_att)
    # Return sets
    return train_set, val_set, test_set


def load_warwick_cross(train_fold, val_fold, test_fold, padding, base_att):

    train_set = ColonCancerBagsCross('./data/CRCHistoPhenotypes_2016_04_28/Classification/',
                                     train_val_idxs=train_fold,
                                     test_idxs=test_fold,
                                     train=True,
                                     shuffle_bag=True,
                                     data_augmentation=True,
                                     padding=padding,
                                     base_att=base_att)

    val_set = ColonCancerBagsCross('./data/CRCHistoPhenotypes_2016_04_28/Classification/',
                                   train_val_idxs=val_fold,
                                   test_idxs=test_fold,
                                   train=True,
                                   shuffle_bag=True,
                                   data_augmentation=True,
                                   padding=padding,
                                   base_att=base_att)

    test_set = ColonCancerBagsCross('./data/CRCHistoPhenotypes_2016_04_28/Classification/',
                                    train_val_idxs=train_fold,
                                    test_idxs=test_fold,
                                    train=False,
                                    shuffle_bag=False,
                                    data_augmentation=False,
                                    padding=padding,
                                    base_att=base_att)

    return train_set, val_set, test_set
