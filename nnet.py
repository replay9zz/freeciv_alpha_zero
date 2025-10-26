from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from alpha_zero_general.NeuralNet import NeuralNet  # type: ignore
from alpha_zero_general.utils import AverageMeter, dotdict  # type: ignore

from freeciv_alpha_zero.freeciv_game import CanonicalBoard
from freeciv_alpha_zero.pytorch.NNet import FreecivNNet

nnet_args = dotdict({
    'lr': 1e-3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': torch.cuda.is_available(),
    'num_channels': 64,
})


class NNetWrapper(NeuralNet):
    def __init__(self, game):
        self.game = game
        self.nnet = FreecivNNet(game, nnet_args)
        self.channels, self.board_h, self.board_w = game.getBoardSize()
        self.action_size = game.getActionSize()
        if nnet_args.cuda:
            self.nnet.cuda()

    # ---- API ----
    def train(self, examples: List[Tuple[object, np.ndarray, float]]):
        optimizer = optim.Adam(self.nnet.parameters(), lr=nnet_args.lr)

        for epoch in range(nnet_args.epochs):
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = max(1, len(examples) // nnet_args.batch_size)
            iterator = tqdm(range(batch_count), desc=f"Epoch {epoch+1}/{nnet_args.epochs}")
            for _ in iterator:
                sample_ids = np.random.randint(len(examples), size=nnet_args.batch_size)
                boards, target_pis, target_vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = np.array([self._board_to_np(b) for b in boards], dtype=np.float32)
                target_pis = torch.FloatTensor(np.array(target_pis))
                target_vs = torch.FloatTensor(np.array(target_vs, dtype=np.float32))
                boards_t = torch.FloatTensor(boards)

                if nnet_args.cuda:
                    boards_t = boards_t.cuda()
                    target_pis = target_pis.cuda()
                    target_vs = target_vs.cuda()

                out_pi, out_v = self.nnet(boards_t)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                pi_losses.update(l_pi.item(), boards_t.size(0))
                v_losses.update(l_v.item(), boards_t.size(0))
                iterator.set_postfix(pi=pi_losses.avg, v=v_losses.avg)

    def predict(self, board):
        self.nnet.eval()
        board_np = self._board_to_np(board)
        board_t = torch.FloatTensor(board_np).unsqueeze(0)
        if nnet_args.cuda:
            board_t = board_t.cuda()
        with torch.no_grad():
            pi, v = self.nnet(board_t)
        return torch.exp(pi).cpu().numpy()[0], v.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        map_location = None if nnet_args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

    # ---- helpers ----
    def _board_to_np(self, board) -> np.ndarray:
        if isinstance(board, CanonicalBoard):
            return board.state.encode(board.perspective)
        # raw board defaults to perspective of player 1
        return board.encode(1)
