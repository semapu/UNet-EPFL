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

Dataset = namedtuple('Dataset', ['image', 'mask', 'coords', 'radius', 'density'])

logger = logging.getLogger(__name__)


# Tool to debug the code in python. Insert where you want stop the execution.
# import IPython
# IPython.embed()


# Change this to load your training data.
def load_training():
    
    trainingFinal_x = []
    trainingFinal_y = []
    
    base_path = "/cvlabdata1/home/marquez/vesicles/"
    # base_path = "/cvlabdata1/home/mas/vesicles/"

    files = [
            "hipp_testing/Bouton 2",
            "hipp_testing/Bouton 3",
            "glutamate/Bouton 1",
            "glutamate/Bouton 2",
            "striatum_training/Bouton 1",
            "striatum_training/Bouton 2",
            "striatum_training/Bouton 3"
            ]

    for file in files:
        logger.info("Loading Training dataset: " + file)
        training_x = np.float32(unet.load_volume(os.path.join(base_path, file, "img.tif")) / 255.0)
        training_y  = np.float32(unet.load_volume(os.path.join(base_path, file, "density.tif")))
        mask = np.float32(unet.load_volume(os.path.join(base_path, file, "mask.tif")))
        training_y[mask == 0] = np.nan
        
        trainingFinal_x.append(training_x)
        trainingFinal_y.append(training_y)
        
    return trainingFinal_x, trainingFinal_y


# Change this to load your training data.
def load_testing():
    files = [
             "hipp_training/Bouton 1",
             "hipp_testing/Bouton 1",
             "glutamate/Bouton 3",
             "glutamate/Bouton 4",
             "striatum_testing/Bouton 1",
             "striatum_testing/Bouton 2",
             "striatum_testing/Bouton 3",
             "striatum_testing/Bouton 4"
            ]

    # base_path = "/cvlabdata1/home/marquez/vesicles/"
    base_path = "/cvlabdata1/home/mas/vesicles/"
    
    collection = []

    for file in files:
        logger.info("Loading and Processing Testing dataset: " + file )
        testing_x = np.float32(unet.load_volume(os.path.join(base_path, file, "img.tif")) / 255.0)

        testing_coords, radius = np.load(os.path.join(base_path, file, "vesicles.npy"), encoding = 'bytes') 
        
        testing_coords = np.vstack(testing_coords)
        radius = np.float_(radius)

        mask = np.float32(unet.load_volume(os.path.join(base_path, file, "mask.tif")))
        
        density  = np.float32(unet.load_volume(os.path.join(base_path, file, "density.tif")))

        collection.append(Dataset(testing_x, mask, testing_coords, radius, density))

    return collection



# REGRESSION - Classic Implementation - Desnity map using emsa.
def training_step(niter, sampler, unet_clsf, optimizer): 

    # De esta manera, junto a todas la funciones de modificación, podemos fijar un minibatch a entrenar
    # Haciendo que así, solo entrene un único elemento y ver como entiende el problema, el slide.
#     niter=0 # Para debugar con DENSITY

    # Get the minibatch
    x, y, w = sampler.get_minibatch(niter)

    y2 = y

    # Convert to pytorch
    x2 = Variable(torch.from_numpy(np.ascontiguousarray(x)).cuda())
    y2 = Variable(torch.from_numpy(np.ascontiguousarray(y)).cuda())
    w2 = Variable(torch.from_numpy(np.ascontiguousarray(w)).cuda())

#     optimizer.zero_grad()

    pred = unet_clsf(x2)[0, 0] # [mini_batch, output_channels, depth, height, width]
    y2 = y2[0] 
    w2 = w2[0] 

    # To debug working with DENSITY
    # if niter == 1000:
    #     import IPython
    #     IPython.embed()
    
    lower_bounds, upper_bounds = emsa.sample_boxes(pred.shape, 
                                                   num_boxes = 4,
                                                   min_radius = 4, 
                                                   max_radius = 8,
                                                   mask = w2.data.cpu().numpy(), 
                                                   random_state = np.random)

    # Using the random baxes to compute the loss
    lb = lower_bounds
    ub = upper_bounds

    losses = []
    aux = w2*pred
    for i in range(lower_bounds.shape[0]):
        aux1 = (aux[lb[i][0]:ub[i][0], lb[i][1]:ub[i][1], lb[i][2]:ub[i][2]]).sum()
        aux2 = (y2[lb[i][0]:ub[i][0], lb[i][1]:ub[i][1], lb[i][2]:ub[i][2]]).sum()
        losses.append((aux1-aux2)**2)

    loss = __builtins__.sum(losses) / len(losses) # Average of the losses onbtained

    optimizer.zero_grad() 
    loss.backward()
    optimizer.step()

    return {"loss": float(loss.data.cpu().numpy())}


def inverse_argwhere(coords, shape, dtype):
    
    res = np.zeros(shape, dtype=dtype)
    intcoords = np.int_(np.round(coords))
    res[intcoords[:, 0], intcoords[:, 1], intcoords[:, 2]] = 1
    return res


def reduce_metrics(results):

    true_sums = np.array([i.true_sums for i in results])
    pred_sums = np.array([i.pred_sums for i in results])
    volumes = np.array([i.volumes for i in results])
    
    error = emsa.diff_vesicles_per_voxel(true_sums, pred_sums, volumes)
    
    return Metrics2(error, true_sums, pred_sums, volumes)



Metrics2 = namedtuple('Metrics', ['error', 'true_sums', 'pred_sums', 'volumes'])



# testing fuction (error)
def test_unet(niter, datasets, unet_clsf, hint_patch_shape):
    
    vesicle_radius = 4
    
    results = []
    
    for dataset in datasets:
        pred = unet.predict_in_blocks(unet_clsf, 
                                      dataset.image, 
                                      hint_patch_shape,
                                      verbose=False)[0]
        
        true_density = dataset.density
        true_sums, pred_sums, volumes = emsa.ersa(true_density, pred, dataset.mask, 1000000, vesicle_radius)
        error = emsa.diff_vesicles_per_voxel(true_sums, pred_sums, volumes)
        
        results.append(Metrics2(error, true_sums, pred_sums, volumes))
    
    total = reduce_metrics(results)
    logger.info("\tError: {:.4g} vesicles/voxel. ".format(total.error))
    
    return {"errors": np.array([i.error for i in results]),
            "true_sums": np.array([i.true_sums[::100] for i in results]),
            "pred_sums": np.array([i.pred_sums[::100] for i in results]),
            "volumes": np.array([i.volumes[::100] for i in results]),
            "total_error": total.error}



def main(hint_patch_shape=(82, 82, 82), # ORIGINAL
# def main(hint_patch_shape=(100, 100, 100),
         learning_rate=1e-4,
         save_path="/cvlabdata2/home/mas/UNet-Density-Test10-New"): 

    logger.info("Loading the testing data...")
    testing_dataset = load_testing() 

    logger.info("Loading the training data...")
    training_x, training_y = load_training()
    
    num_classes = 1 
    logger.info("Creating the network...")
    unet_config = unet.UNetConfig(steps=2,
                                  ndims=3,
                                  num_classes=num_classes,
                                  first_layer_channels=64,
                                  num_input_channels=1,
                                  two_sublayers=True)
    
    in_patch_shape, out_patch_shape = unet_config.in_out_shape(hint_patch_shape)

    unet_clsf = unet.UNetRegressor(unet_config).cuda() 
    
    logger.info("Creating PatchImportanceSampler...")
    sampler = unet.samplers.PatchImportanceSampler(unet_clsf.config,
                                  training_x, training_y, 
                                  in_patch_shape, out_patch_shape,
                                  loss_weights=[1.0 - np.isnan(i) for i in training_y],
                                  transformations = unet.transformation.all_transformations(unet_config.ndims),
                                  mask_func=np.isnan) 

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
    
    
    logger.info("Training the Network...") 
    # unet_trainer.load(200000) # Start again the training (use the last saved status)
    unet_trainer.train(200000, 500)


if __name__ == "__main__":
    unet.config_logger("/cvlabdata2/home/mas/UNet-Density-Test10-New.log")
    main()