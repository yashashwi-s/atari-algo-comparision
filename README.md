Run the files and visulaise the metrics on tensorboard and wandb. Don't forget to change the initials on wandb. 

I saw the third task and instantly knew what to do. I remember thinking why is there a custom DQN and this idea popped up. I wasn't able to implement the complete idea but some work has been done. the research idea is simple - "Multi-Algorithm Neural World Model Comparison"

I have only done the phase one of Paras's version. The plan is to do a complete and detailed study. I'll quickly answer the questions required for this assignment and then tlk about some interesting results.

### Why didn't I choose Atari-Pixels repo and modify that?

I checked the repository and this idea clicked. I tried to run the code but it turned out that it wasn't very optimised to run on low hardwares. As I was clear about my task, I started to build a new repo for no confusions. The larger picture was to merge both the approaches, but this was also clear to me that it is not possible in such time span.

### Why did I pick the particular project?

It instantly clicked with me and I had an amazing idea. After a few google searches I didn't find any recent studies of Neural world model algorithm comaprisions. I thought this could be a good start.

### If I had more compute/time, what would I have done?

If I had more time then completing the study, but by the looks of it this might take some time as the plan is to perform it on a variety of hyperparameters and different games. Currently I am running this for breakout. If I had more compute then definitely running for more timesteps would help. I'll talk in the results section why this more compute thing is very interesting.

### What did I learn in the project?

From implementations I learnt wandb as I wasn't aware of it. I did something similar last year where I used tensorboard but wandb is far superior. I learnt basics of the trifecta algorithms very popular in RL, PPO, DQN and A2C. Although I've used them before this time I had a deeper study.

### What surprised me the most?

The fact that simple projects like these can go wrong in so many ways. I've had a good time exploring why wasn't tesnorboard logging rewards or why wasn't wandb initializing properly. I tried to implement custom callbacks but they didnt work as expected so I used inbuilt functions. This was definitely a memorable experience for me.

### If I had to write a paper on the project, what else needs to be done?

Oh boy, first we need to have multiple runs on different hyperparameters (trying to give each algorithm around the same ground level in each run). The comparision was done for each algorithm given same timesteps, but timesteps wern't a useful constant as different algos have different fps, maybe we can somehow keep the real world training time to be constant for each algorithm, that way it can be a fair comparision. The study also needs to be done on multiple games to get a generalizing result. All this needs to be done only for the phase 1, next we have to do similar steps for the resst of the phases for this to be complete. I think this can be a good study overall.

### Future Scopes of this project

Maybe we can turn this into a proper framework for researchers to test their algorithms on standard games. This in theory could act like a playground and the published tests can be used for future testings. We can try and implement SOTA models and test them against classical models ourselves.

---

## Project Setup

I will compare three popular algorithms PPO, DQN, A2C from `stable-baselines3` and test them against each other. We can also test these algorithms against itself with different starting/learning params and in different games trying to learn into their practical applications. There are some indivdual studies on these, but a good comparrative study in neural world models is absent. This coud be it.
I have taken `BreakoutNoFrameskip-v4` environment from gymnasium. 

## How to run

Change the configurations of each algorithm and the wandb settings. Then train them using their trainer code. You can also test them on the trained models. If you have a special model you'd like them to train on, change the paths in config files. Run tensorboard in ./logs directory to view stats.

## Results

I ran each of the algorithm for 5M steps. DQN took the most time, where as A2C and PPO took almost equal times. Which I later realised was due to the fact that I had 1 env for dqn and 4 for each of ppo and a2c (made my heart cry out). I left the entire process to run for about 20 hours on a GTX 1650. I also learned the difference in methods of explorations and network updates of dqn and ppo makes dqn slower.

![fps chart](https://github.com/user-attachments/assets/1a6c7fe7-916f-4f45-a2c2-4b27c27be516)

For some reason DQN was very slow compared to the rest two, almost took twice the time.

I had capped the timesteps to max at 1000. I soon learned (after a day) that it wasn't a good idea, and my models are performing very poorly because there wasn't enough time for them to learn. This was the results for the three algorithms

![Mean Rewards](https://github.com/user-attachments/assets/e14efafb-f725-4fe7-b4a1-ba4b90565b5c)  ![Algorithms Color Chart](https://github.com/user-attachments/assets/bd1e907e-bda1-4b08-adea-327d082b18fa)

### Rewards

DQN achieved highest mean reward ~10 and episode length ~1000.
A2C plateaued at mean reward ~7.5 and episode length ~980.
PPO gave lowest performance with mean reward ~7.3 and episode length ~980

### Convergence

DQN converged at around 2M, the others at arounf 3-4M.

The reason for this superior performance of dqn may be given to replay buffer, which the other on-policy algorithms lack. I also learnt that Breakout is a classical setting where DQN outperforms naturally so there is a bias to the current extent of study as well.

### Testing the models on 100 episodes



### Another round of training (2M steps) of PPO with 5000 max timesteps




