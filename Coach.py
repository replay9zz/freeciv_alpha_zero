import csv
import logging
import os
import sys
import time
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None

from .Arena import Arena
from .MCTS import MCTS

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.stats_path = getattr(self.args, 'stats_path', None)
        log_dir = getattr(self.args, 'tensorboard_dir', None)
        if SummaryWriter and log_dir:
            self.tb_writer = SummaryWriter(log_dir)
        else:
            self.tb_writer = None
        self.selfplay_episode = 0
        self.live_log_counter = 0
        self.live_log_next = None
        self.live_log_stats = None

    def _log_live_stats(self, force: bool = False) -> None:
        if not self.tb_writer or not self.live_log_stats:
            return
        now = time.time()
        if not force and self.live_log_next is not None and now < self.live_log_next:
            return
        self.live_log_next = now + 0.5
        self.live_log_counter += 1
        stats = self.live_log_stats
        self.tb_writer.add_scalar(
            "0.Live/Total_reward",
            float(stats.get("total_reward", 0.0)),
            self.live_log_counter,
        )
        self.tb_writer.add_scalar(
            "0.Live/Player1_reward",
            float(stats.get("player_rewards", {}).get(1, 0.0)),
            self.live_log_counter,
        )
        self.tb_writer.add_scalar(
            "0.Live/Player2_reward",
            float(stats.get("player_rewards", {}).get(-1, 0.0)),
            self.live_log_counter,
        )
        self.tb_writer.add_scalar(
            "0.Live/Episode_length",
            float(stats.get("episode_length", 0.0)),
            self.live_log_counter,
        )

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
            stats: dict with total_reward, player_rewards, episode_length
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0
        total_reward = 0.0
        player_rewards = {1: 0.0, -1: 0.0}

        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            prev_scores = dict(board.scores) if hasattr(board, "scores") else None
            acting_player = self.curPlayer
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            if prev_scores is not None and hasattr(board, "scores"):
                delta = board.scores.get(acting_player, 0.0) - prev_scores.get(acting_player, 0.0)
                player_rewards[acting_player] += float(delta)
                total_reward += float(delta)

            self._log_live_stats()
            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                stats = {
                    "total_reward": total_reward,
                    "player_rewards": player_rewards,
                    "episode_length": episodeStep,
                }
                examples = [
                    (x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer)))
                    for x in trainExamples
                ]
                return examples, stats

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                log.info("Starting self-play episodes (%d)", self.args.numEps)
                total_rewards = []
                player1_rewards = []
                player2_rewards = []
                episode_lengths = []
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    examples, stats = self.executeEpisode()
                    iterationTrainExamples += examples
                    total_rewards.append(stats["total_reward"])
                    player1_rewards.append(stats["player_rewards"].get(1, 0.0))
                    player2_rewards.append(stats["player_rewards"].get(-1, 0.0))
                    episode_lengths.append(stats["episode_length"])
                    self.live_log_stats = stats
                    self._log_live_stats(force=True)
                    if self.tb_writer:
                        self.selfplay_episode += 1
                        self.tb_writer.add_scalar(
                            "0.Episode/Total_reward",
                            float(stats["total_reward"]),
                            self.selfplay_episode,
                        )
                        self.tb_writer.add_scalar(
                            "0.Episode/Player1_reward",
                            float(stats["player_rewards"].get(1, 0.0)),
                            self.selfplay_episode,
                        )
                        self.tb_writer.add_scalar(
                            "0.Episode/Player2_reward",
                            float(stats["player_rewards"].get(-1, 0.0)),
                            self.selfplay_episode,
                        )
                        self.tb_writer.add_scalar(
                            "0.Episode/Length",
                            float(stats["episode_length"]),
                            self.selfplay_episode,
                        )
                log.info("Finished self-play; examples collected: %d", len(iterationTrainExamples))
                if self.tb_writer and total_rewards:
                    mean_total = float(np.mean(total_rewards))
                    mean_len = float(np.mean(episode_lengths))
                    mean_p1 = float(np.mean(player1_rewards))
                    mean_p2 = float(np.mean(player2_rewards))
                    self.tb_writer.add_scalar("1.Total_reward/1.Total_reward", mean_total, i)
                    self.tb_writer.add_scalar("1.Total_reward/3.Episode_length", mean_len, i)
                    self.tb_writer.add_scalar("1.Total_reward/4.Player1_reward", mean_p1, i)
                    self.tb_writer.add_scalar("1.Total_reward/5.Player2_reward", mean_p2, i)
                    self.tb_writer.flush()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            log.info("Starting network training on %d examples", len(trainExamples))
            self.nnet.train(trainExamples)
            log.info("Finished network training")
            nmcts = MCTS(self.game, self.nnet, self.args)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
                          lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            accepted = False
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
                accepted = True

            self._log_iteration_stats(i, len(trainExamples), pwins, nwins, draws, accepted)
        if self.tb_writer:
            self.tb_writer.close()

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True

    def _log_iteration_stats(self, iteration, example_count, pwins, nwins, draws, accepted):
        if self.stats_path:
            folder = os.path.dirname(self.stats_path)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

            file_exists = os.path.isfile(self.stats_path)
            with open(self.stats_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['iteration', 'examples', 'prev_wins', 'new_wins', 'draws', 'accepted'])
                writer.writerow([iteration, example_count, pwins, nwins, draws, int(accepted)])

        if self.tb_writer:
            self.tb_writer.add_scalar("train/examples", example_count, iteration)
            self.tb_writer.add_scalar("arena/prev_wins", pwins, iteration)
            self.tb_writer.add_scalar("arena/new_wins", nwins, iteration)
            self.tb_writer.add_scalar("arena/draws", draws, iteration)
            total = pwins + nwins + draws
            if total > 0:
                self.tb_writer.add_scalar("arena/new_win_rate", nwins / total, iteration)
            self.tb_writer.add_scalar("arena/accepted", int(accepted), iteration)
            self.tb_writer.flush()
