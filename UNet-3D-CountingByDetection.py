import logging
import sys
import os.path
from collections import namedtuple
from itertools import chain

import emsa # Python code to generate random boxes

import numpy as np
import pandas as pd
import scipy
from scipy import ndimage
from scipy import ndimage as ndi
from scipy.misc import imsave
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import optim

import unet
from unet.utils import BinCounter

Dataset = namedtuple('Dataset', ['image', 'mask', 'coords', 'radius'])

logger = logging.getLogger(__name__)

# Tool to debug the code in python. Insert where you want stop the execution.
# import IPython
# IPython.embed()

def vesicle_3D_dilation(path):

    mask = np.float32(unet.load_volume(os.path.join(path, "mask.tif")))
    vesicles, radius = np.load(os.path.join(path, "vesicles.npy"), encoding = 'bytes')
    vesicles = np.vstack(vesicles)
    radius = np.float_(radius)

    # Load of the centers
    ones = np.int64(np.round(vesicles))

    # Centers activation with Advanced Indexing
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[ones[:, 0], ones[:, 1], ones[:, 2]] = 1

    # 3D dilation using euclidean distances.
    distance = ndimage.distance_transform_edt(1 - labels) # Distance to the nearest black point. We need the distance to the white.
    labels2 = np.int64(distance < 1.4) # The threshold by the centers of the vesicles.
    labels = np.int64(distance < 1.4) # Threshold by the area of uncertainty.

    # '1' is the center and '2' the area of uncertainty
    labels[labels == 1] = 2
    labels[labels2 == 1] = 1

    #Area of uncertainty in the initial mask of the vesicle.
    labels[mask == 0] = 2

    return labels

# Change this to load your training data.
def load_training():

    trainingFinal_x = []
    trainingFinal_y = []

    # base_path = "/cvlabdata1/home/marquez/vesicles/"
    base_path = "/cvlabdata1/home/mas/vesicles/"

    files = ["hipp_testing/Bouton 2",
            "hipp_testing/Bouton 3",
            "glutamate/Bouton 1",
            "glutamate/Bouton 2",
            "striatum_training/Bouton 1",
            "striatum_training/Bouton 2",
            "striatum_training/Bouton 3"]

    for file in files:
        logger.info("Computing 3D dilation in Training dataset: " + file)
        training_x = np.float32(unet.load_volume(os.path.join(base_path, file, "img.tif")) / 255.0)
        training_y = vesicle_3D_dilation(os.path.join(base_path, file))
        trainingFinal_x.append(training_x)
        trainingFinal_y.append(training_y)

    return trainingFinal_x, trainingFinal_y

# Change this to load your training data.
def load_testing():
    files = ["hipp_training/Bouton 1",
             "hipp_testing/Bouton 1",
             "glutamate/Bouton 3",
             "glutamate/Bouton 4",
             "striatum_testing/Bouton 1",
             "striatum_testing/Bouton 2",
             "striatum_testing/Bouton 3",
             "striatum_testing/Bouton 4"]

    # base_path = "/cvlabdata1/home/marquez/vesicles/"
    base_path = "/cvlabdata1/home/mas/vesicles/
    
    collection = []

    for file in files:
        logger.info("Computing 3D dilation in Testing dataset: " + file )        
        testing_x = np.float32(unet.load_volume(os.path.join(base_path, file, "img.tif")) / 255.0)


        testing_coords, radius = np.load(os.path.join(base_path, file, "vesicles.npy"), encoding = 'bytes')

        testing_coords = np.vstack(testing_coords)
        radius = np.float_(radius)

        mask = np.float32(unet.load_volume(os.path.join(base_path, file, "mask.tif")))

        collection.append(Dataset(testing_x, mask, testing_coords, radius))

    return collection

# Nonmaxima-supression to abtain the centers in the predictions.
def nonmaxima_suppression(img, return_mask=True):
    # smooth_img = img # ndi.gaussian_filter(img, 1)
    dilated = ndi.grey_dilation(img, (5,) * img.ndim)
    argmaxima = np.logical_and(img == dilated, img > 0.5)

    argwhere = np.argwhere(argmaxima)

    if not return_mask:
        return argwhere

    return argwhere, argmaxima


def invfreq_lossweights(labels, num_classes):
    bc = BinCounter(num_classes + 1)
    for labels_i in labels:
        bc.update(labels_i)
    class_weights = 1.0 / (num_classes * bc.frequencies)[:num_classes]
    class_weights = np.hstack([class_weights, 0])
    return np.float32(class_weights)


# CLASSIFIER - Ninary cross entropy loss.
def training_step(niter, sampler, unet_clsf, optimizer):

    # Get the minibatch
    x, y, w = sampler.get_minibatch(niter)

    y2 = y

    # Convert to pytorch
    x2 = Variable(torch.from_numpy(np.ascontiguousarray(x)).cuda())
    y2 = Variable(torch.from_numpy(np.ascontiguousarray(y)).cuda())
    w2 = Variable(torch.from_numpy(np.ascontiguousarray(w)).cuda())

    optimizer.zero_grad()
    loss = unet_clsf.loss(x2, y2, w2)
    loss.backward()
    optimizer.step()

    return {"loss": float(loss.data.cpu().numpy())}


def inverse_argwhere(coords, shape, dtype):

    res = np.zeros(shape, dtype=dtype)
    intcoords = np.int_(np.round(coords))
    res[intcoords[:, 0], intcoords[:, 1], intcoords[:, 2]] = 1
    return res


def precision_and_recall(testing_coords, pred_coords, match_distance):
    w = scipy.spatial.distance_matrix(testing_coords, pred_coords)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(w)

    res = []
    for i in range(row_ind.shape[0]):
        if w[row_ind[i], col_ind[i]] <= match_distance:
            res.append(w[row_ind[i], col_ind[i]])

    precision = len(res) / len(pred_coords)
    recall = len(res) / len(testing_coords)

    return precision, recall, len(res), len(pred_coords), len(testing_coords)


Metrics = namedtuple('Metrics', ['precision', 'recall', 'tp', 'tp_fp', 'tp_fn',
                                 'error', 'true_sums', 'pred_sums', 'volumes'])


# Structuration of the results in testing.
def reduce_metrics(results):

    tp = sum(i.tp for i in results)
    tp_fp = sum(i.tp_fp for i in results)
    tp_fn = sum(i.tp_fn for i in results)
    true_sums = np.array([i.true_sums for i in results])
    pred_sums = np.array([i.pred_sums for i in results])
    volumes = np.array([i.volumes for i in results])

    precision = tp / tp_fp
    recall = tp / tp_fn

    error = emsa.diff_vesicles_per_voxel(true_sums, pred_sums, volumes)

    return Metrics(precision, recall, tp, tp_fp, tp_fn,
                   error, true_sums, pred_sums, volumes)


# testing fuction
def test_unet(niter, datasets, unet_clsf, hint_patch_shape):

    vesicle_radius = 4

    results = []

    for dataset in datasets:
        pred = unet.predict_in_blocks(unet_clsf, dataset.image, hint_patch_shape,
                                      output_function=unet_clsf.probability_output,
                                      verbose=False)[1]
        
        pred_coords, pred_y = nonmaxima_suppression(pred * dataset.mask, True)

        testing_y = inverse_argwhere(dataset.coords, dataset.image.shape, dtype=np.float32)

        soft_testing_y = ndi.gaussian_filter(testing_y, 2)
        soft_pred_y = ndi.gaussian_filter(np.float32(pred_y), 2)

        true_sums, pred_sums, volumes = emsa.ersa(soft_testing_y, soft_pred_y, dataset.mask, 1000000, vesicle_radius)
        error = emsa.diff_vesicles_per_voxel(true_sums, pred_sums, volumes)

        precision, recall, tp, tp_fp, tp_fn = precision_and_recall(dataset.coords, pred_coords,
                                                                   match_distance=vesicle_radius)

        results.append(Metrics(precision, recall, tp, tp_fp, tp_fn,
                               error, true_sums, pred_sums, volumes))

    total = reduce_metrics(results)
    logger.info("\tPrecision: {:.3f}. "
                "Recall: {:.3f}. "
                "Error: {:.3e} elements/voxel.".format(total.precision, total.recall, total.error))

    return {"precisions": np.array([i.precision for i in results]),
            "recalls": np.array([i.recall for i in results]),
            "errors": np.array([i.error for i in results]),
            "true_sums": np.array([i.true_sums[::100] for i in results]),
            "pred_sums": np.array([i.pred_sums[::100] for i in results]),
            "volumes": np.array([i.volumes[::100] for i in results]),
            "total_precision": total.precision,
            "total_recall": total.recall,
            "total_error": total.error}

def main(hint_patch_shape=(82, 82, 82),
         learning_rate=1e-4,
         save_path="/cvlabdata2/home/mas/UNet-3D-Test1"): 
    
    logger.info("Loading the testing data...")
    testing_dataset = load_testing()

    logger.info("Loading the training data...")
    training_x, training_y = load_training()

    logger.info("Creating the Network...")
    num_classes = 2
    unet_config = unet.UNetConfig(steps=2,
                                  ndims=3,
                                  num_classes=num_classes,
                                  first_layer_channels=64,
                                  num_input_channels=1,
                                  two_sublayers=True)

    class_weights = invfreq_lossweights(training_y, num_classes)
    loss_weights = [class_weights[i] for i in training_y]

    in_patch_shape, out_patch_shape = unet_config.in_out_shape(hint_patch_shape)

    unet_clsf = unet.UNetClassifier(unet_config).cuda() # .cuda() moves the network to GPU
    
    logger.info("Creating sampler and the optimizer...")
    sampler = unet.samplers.PatchImportanceSampler(unet_clsf.config,
                                  training_x, training_y,
                                  in_patch_shape, out_patch_shape,
                                  loss_weights=loss_weights,
                                  transformations = unet.transformation.all_transformations(unet_config.ndims),
                                  mask_func=lambda x: x == num_classes)

    optimizer = unet.setup.optim.Adam(unet_clsf.parameters(), lr=learning_rate)

    logger.info("Creating trainer...")    
    unet_trainer = unet.trainer.Trainer(lambda niter : training_step(niter, sampler, unet_clsf, optimizer),
                                save_every=5000,
                                save_path=save_path,
                                managed_objects=unet.trainer.managed_objects({"network": unet_clsf,
                                                                              "optimizer": optimizer}),
                                test_function=lambda niter: test_unet(niter,
                                                                      testing_dataset,
                                                                      unet_clsf,
                                                                      hint_patch_shape),
                                test_every=5000)
    # unet_trainer.load(200000) # Start again the training (use the last saved status)
    logger.info("Training the Network...")
    unet_trainer.train(200000, 500)

    
if __name__ == "__main__":
    unet.config_logger("/cvlabdata2/home/mas/UNet-3D-Test1.log")
    main()