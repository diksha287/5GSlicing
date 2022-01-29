import torch
import torch.nn as nn
import numpy as np
from random import randint as ri

def multiDimBatchArgmax(arr):
    '''
    For a tensor of shape (N, D1, D2, ....) return a list of N tuples(indices)
    for the max element for each element in the batch.
    '''
    a = arr.cpu().numpy()
    indices = [np.unravel_index(np.argmax(r), r.shape) for r in a]
    return indices

def chWidthFromChNum(chNum):
    if chNum < 29: return 20
    elif chNum < 43: return 40
    elif chNum < 50: return 80
    else : return 160

class ModelHelper():
    '''
    Helper class that trains the model and returns actions, targets for a given state
    '''
    def __init__(self, device, outdir, resume_from, batch_size = 4, epsilon = 0.1):
        self.device = device
        self.outdir = outdir
        self.idx = 0

    def getActionTuple(self, obs, action):
        '''
        Returns a vector with values {0,1,2} using the trained model and
        last observation and action, The vector decides whether a variable
        should be decreased, increased or remain constant.
        '''
        return tuple([ri(0, 2) for i in range(12)])

    def getActionFromActionTuple(self, action_tuple, action):
        '''
        Returns a new action(that can be put into simulations) from action tuple
        increments or decrements action values based on the action tuple
        0 --> -1
        1 --> do nothing
        2 --> +1
        if value is min no decrement is possible and
        if value is max no increment is possible
        '''
        action_names = ["chNum", "gi", "mcs", "txPower"]
        max_min_dict = {
            "chNum":[0, 52],
            "gi":[0, 2],
            "mcs" : [0, 11],
            "txPower" : [1, 20]
        }
        for j, action_name in enumerate(action_names):
            for i in range(3):
                if action_tuple[3*j + i] == 0 :
                    action[action_name][i] = max(action[action_name][i] - 1, max_min_dict[action_name][0])
                elif action_tuple[3*j + i] == 2 :
                    action[action_name][i] = min(action[action_name][i] + 1, max_min_dict[action_name][1])
        return action

    def saveObsActionFeaturesInMemory(self, obs, action, action_tuple, obs_new):
        featA, featB, featC = self.getInputFeaturesFromObservation(obs)
        actA, actB, actC = torch.unbind(self.convertActionToTensor(action))

        featA = torch.cat([featA, actA])
        featB = torch.cat([featB, actB])
        featC = torch.cat([featC, actC])
        #Todo : Normalize

        x = torch.cat([featA, featB, featC]).unsqueeze(0)
        real_target = self.getTarget(obs_new, action)
        train_data = {
            'x': x,
            'action_tuple': action_tuple,
            'real_target': real_target,
        }
        torch.save(train_data,
                    self.outdir + str(self.idx) + '.pt')
        self.idx += 1

    def getTarget(self, obs, action):
        '''
        returns a function of the following
        - Latency
        - Error probability (tx-rx)/tx
        - Transmission power
        - Spectral efficiency (sum(rxpackets)/time)/bandwidth
        We want to maximize the target value.
        '''

        obsTensors = self.convertObsToTensors(obs)
        mean_error_probA = torch.mean(1 - obsTensors[0][1]/obsTensors[0][0])
        mean_error_probB = torch.mean(1 - obsTensors[1][1]/obsTensors[1][0])
        mean_error_probC = torch.mean(1 - obsTensors[2][1]/obsTensors[2][0])
        return ean_error_probA + mean_error_probB + mean_error_probC
        
#         mean_latencyA = torch.mean(obsTensors[0][3])
#         mean_error_probA = torch.mean(1 - obsTensors[0][2]/obsTensors[0][1])
#         targetA = -mean_latencyA - mean_error_probA                               #eMBB

#         mean_latencyB = torch.mean(obsTensors[1][3])
#         mean_error_probB = torch.mean(1 - obsTensors[1][2]/obsTensors[1][1])
#         txPowerB = action["txPower"][1]     #Todo : multiply with numStationsB?
#         targetB = - mean_latencyB - mean_error_probB - txPowerB                   #Minimize power used as well mMTC

#         mean_latencyC = torch.mean(obsTensors[2][3])
#         mean_error_probC = torch.mean(1 - obsTensors[2][2]/obsTensors[2][1])
#         targetC = -10*mean_latencyC - mean_error_probC                               #Focus more on latency URLLC

#         throughput = sum([sum(obsTensors[i][2]) * 1472 * 8 for i in range(3)])/1000000.0  #1472 is payload size, 8 bits #Todo : Divide by sim time ?
#         totalBandWidth = sum([chWidthFromChNum(action["chNum"][i]) for i in range(3)])
#         se = throughput/totalBandWidth

#         return targetA + targetB + targetC + se

    def getInputFeaturesFromObservation(self, obs):
        obsTensors = self.convertObsToTensors(obs)
        obsTensors = self.normalizeObs(obsTensors)
        funcs = [torch.mean, torch.var, torch.min, torch.max]
        featuresA = torch.stack([func(elem) for elem in obsTensors[0] for func in funcs])
        featuresB = torch.stack([func(elem) for elem in obsTensors[1] for func in funcs])
        featuresC = torch.stack([func(elem) for elem in obsTensors[2] for func in funcs])

        #print(featuresA.shape)  #(20-d vector)
        return featuresA, featuresB, featuresC

    def convertObsToTensors(self, obs):
       # drA = torch.nan_to_num(torch.Tensor(obs["SliceA"][0]).float(), nan=0.0)      #datarate
        tpA = torch.nan_to_num(torch.Tensor(obs["SliceA"][1]).float(), nan=1.0)      #txpackets
        rpA = torch.nan_to_num(torch.Tensor(obs["SliceA"][2]).float(), nan=0.0)      #rxpackets
        lA = torch.nan_to_num(torch.Tensor(obs["SliceA"][3]).float(), nan=5000.0)    #latency
        #rPowA = torch.nan_to_num(torch.Tensor(obs["SliceA"][4]).float(), nan=0.0)    #rxPower

        #drB = torch.nan_to_num(torch.Tensor(obs["SliceB"][0]).float(), nan=0.0)
        tpB = torch.nan_to_num(torch.Tensor(obs["SliceB"][1]).float(), nan=1.0)
        rpB = torch.nan_to_num(torch.Tensor(obs["SliceB"][2]).float(), nan=0.0)
        lB = torch.nan_to_num(torch.Tensor(obs["SliceB"][3]).float(), nan=5000.0)
        #rPowB = torch.nan_to_num(torch.Tensor(obs["SliceB"][4]).float(), nan=0.0)


       # drC = torch.nan_to_num(torch.Tensor(obs["SliceC"][0]).float(), nan=0.0)
        tpC = torch.nan_to_num(torch.Tensor(obs["SliceC"][1]).float(), nan=1.0)
        rpC = torch.nan_to_num(torch.Tensor(obs["SliceC"][2]).float(), nan=0.0)
        lC = torch.nan_to_num(torch.Tensor(obs["SliceC"][3]).float(), nan=5000.0)
        #rPowC = torch.nan_to_num(torch.Tensor(obs["SliceC"][4]).float(), nan=0.0)

        return [(tpA, rpA, lA),
                (tpB, rpB, lB),
                (tpC, rpC, lC)]

    def normalizeObs(self, obsTensors):
        (tpA, rpA, lA),(tpB, rpB, lB),(tpC, rpC, lC) = obsTensors
       # drA = drA / 100
        tpA = tpA / 100000
        rpA = rpA / 100000
        lA = lA / 1000
        #rPowA = rPowA / 20

       # drB = drB / 100000
        tpB = tpB / 100
        rpB = rpB / 100
        lB = lB / 1000
        #rPowB = rPowB / 20

      #  drC = drC / 100
        tpC = tpC / 100000
        rpC = tpC / 100000
        lC = lC / 1000
        #rPowC = rPowC / 20
        return [(tpA, rpA, lA),
                (tpB, rpB, lB),
                (tpC, rpC, lC)]

    def convertActionToTensor(self, action):
        '''
        Convert to tensor and normalize
        '''
        chNum = torch.Tensor(action["chNum"]).float() / 60
        gi = torch.Tensor(action["gi"]).float()
        mcs = torch.Tensor(action["mcs"]).float() / 10
        txPower = torch.Tensor(action["txPower"]).float() / 20

        return torch.stack([chNum, gi, mcs, txPower]).transpose(0, 1)
