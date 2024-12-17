import os
import logging
import time
from pathlib import Path
import sys
sys.path.append(os.path.relpath("./criterion.py"))
sys.path.append(os.path.relpath("./convAE.py"))
sys.path.append(os.path.relpath("./dataloader.py"))

import criterion as cr
import convAE as conv
import dataloader as dl
#import standardization as std

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Callable, Optional, Tuple
import h5py

#torch.set_default_dtype(torch.float32)

def TrainEpoch(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        trainDataloader,
        validationDataloader,
        profiler: Optional[torch.profiler.profile] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,       
):
    '''Train over single epoch'''

    trainLoss = 0
    samplesSeen = 0
    startTime = time.time()
    model.train()
    for batch_idx, data in enumerate(trainDataloader):
        print(f'Batch -- {batch_idx}')
        bolometerOut = data[:,:,0,:]
        witnesses = data[:,:,1:,:]

        optimizer.zero_grad(set_to_none=True) # clear gradient
        # forward step
        with torch.autocast("cuda", enabled=scaler is not None):
            noisePrediction = model(witnesses)
            loss = criterion(noisePrediction, bolometerOut)
            
        trainLoss += loss.item() * len(witnesses) # why multiply with length of witnesses?
        samplesSeen += len(witnesses)
        
        # backpropagation
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        if profiler is not None:
            profiler.stop()
            
        endTime = time.time()
        duration = endTime - startTime
        trainRate = samplesSeen / duration
        trainLoss /= samplesSeen
        
        logging.info(
            f'Duration {duration:.2f}s, {trainRate:.1f} samples/s'
        )

        trainLog = f'Training loss: {trainLoss:.4e}'
        print(f'Training loss: {trainLoss:.4e}')
        
        # validation performance
        if validationDataloader is not None:
            validationLoss = 0
            samplesSeen = 0
            
            model.eval()
            with torch.no_grad():
                for data in validationDataloader:
                    bolometerOut = data[:,:,0,:]
                    witnesses = data[:,:,1:,:]

                    noisePrediction = model(witnesses)
                    loss = criterion(noisePrediction, bolometerOut)
                    
                    validationLoss += loss.item() * len(witnesses)
                    samplesSeen += len(witnesses)
                
            validationLoss /= samplesSeen
            trainLog += f' -- Validation loss: {validationLoss:.4e}'
        else:
            validationLoss = None
            
        logging.info(trainLog)
        return trainLoss, validationLoss, duration, trainRate
    
def Train(
    outputDirectory: str,
    # data
    stdDataPath: str,
    # witnessOut: np.ndarray,
    # boloOut: np.ndarray,
    # kernelLength: float,
    # kernelStride: float,
    sampleRate: float,
    # validationData: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    # pre-processing
    freqLow= None,
    freqHigh=None,
    # optimization
    batchSize: int = 32,
    maxEpochs: int = 40,
    initWeights: Optional[Path] = None,
    learningRate: float = 1e-3,
    weightDecay: float = 0.0,
    patience: Optional[int] = None,
    factor: float = 0.1,
    earlyStop: int = 20,
    #criterion
    fftLength: float = 2,
    # misc
    device: Optional[str] = None,
    profile: bool = False, # Check profiler docs
    useAmp: bool = False, # Check use_amp docs
) -> float:
    '''Train for multiple epochs'''
    
    os.makedirs(outputDirectory, exist_ok=True)
    outputDirectoryPath = Path(outputDirectory)
    
    device = device or "cpu"
    
    logging.info("Preprocessing")
    print("Preprocessing")
    # Standardize witness output and bolometer output
    # Bandpass - bolometer already goes through Bessel bandpass, no implementation for now
    
    dataset = dl.DatasetFromNPZ(npzFile='', batchSize=32, preprocess=False, standardizedDataPath=stdDataPath)
    trainDataset, valDataset, testDataset = dataset.split_data(trainRatio=0.7, valRatio=0.2)

    trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True)
    validationLoader = DataLoader(valDataset, batch_size=1, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)
    # Standardizing training and validation data
    # witnessScaler = std.Standardizer()
    # bolometerScaler = std.Standardizer()

    # witnessScaled = witnessScaler.Fit(witnessOut)
    # bolometerScaled = bolometerScaler.Fit(bolometerOut)

    # Model, loss function, optimizer, and lr scheduler
    print("Defining model")
    model = conv.ModelAE(inChannels=1, numWitnesses=15, kernelLength=100)
    model.to(device)
    
    if initWeights is not None:
        # Use weights from previous run if they exist
        if initWeights.is_dir():
            initWeights = initWeights / "weights.pt"
            
        logging.debug(f'Initializing model weights from checkpoint "{initWeights}"')
        model.load_state_dict(torch.load(initWeights))    
    logging.info(model)
    
    logging.info("Initializing loss and optimizer")
    print("Defining criteria")
    criterion = cr.PSDLoss(
        sampleRate,
        fftLength = fftLength,
        overlap = 0.0,
        device = device,
        freqLow = freqLow,
        freqHigh = freqHigh
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learningRate, weight_decay=weightDecay
    )
    if patience is not None:
        lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience = patience,
            factor = factor,
            threshold = 0.0001,
            min_lr = learningRate * factor **2,
            verbose = True,
        )
    
    # training loop
    scaler = None
    if useAmp and device.startswith("cuda"):
        scaler = torch.amp.GradScaler()
    elif useAmp:
        logging.warning("'useAmp' set to True but no cuda device to use, no action taken")
    
    torch.backends.cudnn.benchmark = True
    bestValidLoss = float("inf")
    sinceLastImprovement = 0
    weightsPath = outputDirectoryPath / "weight.pt"
    trainingHistory = {"trainLoss": [], "validationLoss": []}
    
    logging.info("Start training loop")
    for epoch in range(maxEpochs):
        if epoch == 0 and profile:
            profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=10),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    outputDirectoryPath / "profiler"
                ),
            )
            profiler.start()
        else:
            profiler = None
        
        logging.info(f'Epoch {epoch + 1} / {maxEpochs}')
        print(f'Epoch {epoch + 1} / {maxEpochs}')
        trainLoss, validationLoss, duration, trainRate = TrainEpoch(
            model,
            optimizer,
            criterion,
            trainLoader,
            validationLoader,
            profiler,
            scaler,
        )
        trainingHistory["TrainLoss"].append(trainLoss)
        if validationLoss is not None:
            trainingHistory["validationLoss"].append[validationLoss]

            if patience is not None:
                lrScheduler.step(validationLoss)

            if validationLoss < bestValidLoss:
                logging.debug(
                    "Improvement upon best validation loss, saving model weights"
                )
                bestValidLoss = validationLoss
                torch.save(model.state_dict(), weightsPath)
                sinceLastImprovement = 0
            else:
                sinceLastImprovement += 1
                if sinceLastImprovement >= earlyStop:
                    logging.info(
                        f"Validation loss did not improve for {earlyStop} epochs, stopping loop early"
                    )
                    break
        else:
            # If validation data not available, save most recent model state
            torch.save(model.state_dict(), weightsPath)

    # Load in best weight set 
    model.load_stat_dict(torch.load(weightsPath))

    # Model analysis, TODO: define analyzeModel
    # logging.info("Performing analyses of trained model")
    # gradients, coherences = analyzeModel(trainData, model, sampleRate)

    # Saving training history
    with h5py.File(outputDirectoryPath / "TrainingHistory.h5", "w") as f:
        for key, value in trainingHistory.items():
            f.create_dataset(key, data=value)