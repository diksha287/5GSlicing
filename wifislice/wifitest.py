from ns3gym import ns3env
from tqdm.notebook import tqdm
from Model import ModelHelper
import torch


def runSimulation(
    start_sim: bool,
    iterations: int,
    sim_time: int,
    resume_from: str,
    outdir: str,
    debug: bool,
    batch_size = 4,
    epsilon = 0.1
):
    """
    Train the model.
    """

    port = 5555
    seed = 0
    step_time = 1.0       # Do not change
    simArgs = {"--simulationTime": sim_time,
            "--testArg": 123}
    
    env = ns3env.Ns3Env(port=port, stepTime=step_time, startSim=start_sim, simSeed=seed, simArgs=simArgs, debug=debug)
    # simpler:
    #env = ns3env.Ns3Env()
    env.reset()

    if debug:
        ob_space = env.observation_space
        ac_space = env.action_space
        print("Obseration space", ob_space)
        print("Action space", ac_space)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    helper = ModelHelper(device, outdir, resume_from, batch_size=batch_size, epsilon=epsilon)
    try:

        for currIt in tqdm(range(iterations)):
            obs = env.reset()
            stepIdx = 0
            done = False
            pbar = tqdm(total=sim_time // step_time)
            while not done:
                #Update calls start from 2 secs.
                if stepIdx < 2:
                    action = env.action_space.sample()
                    obs_cur, _, done, _ = env.step(action)
                    loss = None

                else:
                    actionTuple = helper.getActionTuple(obs_prev, action)
                    action = helper.getActionFromActionTuple(actionTuple, action)
                    obs_cur, _, done, _ = env.step(action)

                    #sas' (reward is a funciton of s in this case)
                    helper.saveObsActionFeaturesInMemory(obs_prev, action, actionTuple, obs_cur)

                obs_prev = obs_cur
                stepIdx += 1
                pbar.set_postfix({'Idx' : stepIdx})
                pbar.update(1)
            pbar.close()

    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    finally:
        print("Closing env")
        del env
        #env.close()
        print("Done")
