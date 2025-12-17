from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

try:
    from freeciv_alpha_zero.NeuralNet import NeuralNet  # type: ignore
    from freeciv_alpha_zero.utils import AverageMeter, dotdict  # type: ignore
    from freeciv_alpha_zero.pytorch.NNet import FreecivNNet  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    from ..NeuralNet import NeuralNet
    from ..utils import AverageMeter, dotdict
    from ..pytorch.NNet import FreecivNNet

from .game import CanonicalBoard

force_cpu = os.environ.get("FREECIV_FORCE_CPU", "").strip()
use_cuda = torch.cuda.is_available() and not force_cpu

nnet_args = dotdict({
    'lr': 1e-3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': use_cuda,
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
            print("[nnet] Using CUDA")
        else:
            print("[nnet] Using CPU")

    # ---- API ----
    def train(self, examples: List[Tuple[object, np.ndarray, float]]):
        optimizer = optim.Adam(self.nnet.parameters(), lr=nnet_args.lr)
        print(f"[nnet] Train start: epochs={nnet_args.epochs}, batches_per_epoch={max(1, len(examples) // nnet_args.batch_size)}")

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
        checkpoint = torch.load(filepath, map_location=map_location, weights_only=True)
        state_dict = checkpoint['state_dict']
        aligned_state = self._align_state_dict_shapes(state_dict)
        self.nnet.load_state_dict(aligned_state)

    def _align_state_dict_shapes(self, state_dict):
        """
        Some older checkpoints were trained before additional input channels and actions were added.
        To keep those checkpoints usable, we expand their tensors to the current shapes and let
        the new positions fall back to the freshly-initialized weights.
        """
        model_state = self.nnet.state_dict()
        updated = dict(state_dict)
        notes = []

        # Expand first conv layer if checkpoint has fewer input channels.
        if 'conv1.weight' in state_dict:
            ck = state_dict['conv1.weight']
            tgt = model_state['conv1.weight']
            if ck.shape != tgt.shape:
                if (
                    ck.shape[0] == tgt.shape[0]
                    and ck.shape[2:] == tgt.shape[2:]
                    and ck.shape[1] <= tgt.shape[1]
                ):
                    new_w = tgt.clone()
                    new_w[:, :ck.shape[1], :, :] = ck
                    updated['conv1.weight'] = new_w
                    notes.append(f"conv1.weight {ck.shape} -> {tgt.shape}")
                else:
                    raise RuntimeError(f"Incompatible conv1.weight shape: checkpoint {ck.shape}, expected {tgt.shape}")

        # Expand policy head outputs when the checkpoint had fewer actions.
        if 'policy_head.weight' in state_dict:
            ck = state_dict['policy_head.weight']
            tgt = model_state['policy_head.weight']
            if ck.shape != tgt.shape:
                if ck.shape[1] == tgt.shape[1] and ck.shape[0] <= tgt.shape[0]:
                    new_w = tgt.clone()
                    new_w[:ck.shape[0], :] = ck
                    updated['policy_head.weight'] = new_w
                    notes.append(f"policy_head.weight {ck.shape} -> {tgt.shape}")
                else:
                    raise RuntimeError(f"Incompatible policy_head.weight shape: checkpoint {ck.shape}, expected {tgt.shape}")

        if 'policy_head.bias' in state_dict:
            ck = state_dict['policy_head.bias']
            tgt = model_state['policy_head.bias']
            if ck.shape != tgt.shape:
                if ck.shape[0] <= tgt.shape[0]:
                    new_b = tgt.clone()
                    new_b[:ck.shape[0]] = ck
                    updated['policy_head.bias'] = new_b
                    notes.append(f"policy_head.bias {ck.shape} -> {tgt.shape}")
                else:
                    raise RuntimeError(f"Incompatible policy_head.bias shape: checkpoint {ck.shape}, expected {tgt.shape}")

        # Multi-head models: allow expanding the econ head when new econ actions are inserted.
        if 'policy_econ.weight' in state_dict and 'policy_econ.weight' in model_state:
            ck = state_dict['policy_econ.weight']
            tgt = model_state['policy_econ.weight']
            if ck.shape != tgt.shape:
                if ck.shape[1] == tgt.shape[1] and ck.shape[0] < tgt.shape[0] and ck.shape[0] >= 2:
                    new_w = tgt.clone()
                    # Preserve research rows (prefix) and map the old "pass" row to the new last row.
                    new_w[: ck.shape[0] - 1, :] = ck[: ck.shape[0] - 1, :]
                    new_w[-1, :] = ck[-1, :]
                    updated['policy_econ.weight'] = new_w
                    notes.append(f"policy_econ.weight {ck.shape} -> {tgt.shape}")
                else:
                    raise RuntimeError(f"Incompatible policy_econ.weight shape: checkpoint {ck.shape}, expected {tgt.shape}")

        if 'policy_econ.bias' in state_dict and 'policy_econ.bias' in model_state:
            ck = state_dict['policy_econ.bias']
            tgt = model_state['policy_econ.bias']
            if ck.shape != tgt.shape:
                if ck.shape[0] < tgt.shape[0] and ck.shape[0] >= 2:
                    new_b = tgt.clone()
                    new_b[: ck.shape[0] - 1] = ck[: ck.shape[0] - 1]
                    new_b[-1] = ck[-1]
                    updated['policy_econ.bias'] = new_b
                    notes.append(f"policy_econ.bias {ck.shape} -> {tgt.shape}")
                else:
                    raise RuntimeError(f"Incompatible policy_econ.bias shape: checkpoint {ck.shape}, expected {tgt.shape}")

        if notes:
            print("Adjusted checkpoint for newer model shape:", "; ".join(notes))
        return updated

    # ---- helpers ----
    def _board_to_np(self, board) -> np.ndarray:
        if isinstance(board, CanonicalBoard):
            return board.state.encode(board.perspective)
        # raw board defaults to perspective of player 1
        return board.encode(1)
