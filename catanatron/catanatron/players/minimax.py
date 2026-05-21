import time
import random
from typing import Any

from catanatron.game import Game
from catanatron.models.player import Player, Color
from catanatron.players.tree_search_utils import expand_spectrum, list_prunned_actions
from catanatron.players.value import (
    DEFAULT_WEIGHTS,
    get_value_fn,
)
from catanatron.models.enums import Action, ActionType


ALPHABETA_DEFAULT_DEPTH = 2
MAX_SEARCH_TIME_SECS = 200


class AlphaBetaPlayer(Player):
    """
    Player that executes an AlphaBeta Search where the value of each node
    is taken to be the expected value (using the probability of rolls, etc...)
    of its children. At leafs we simply use the heuristic function given.

    NOTE: More than 3 levels seems to take much longer, it would be
    interesting to see this with prunning.
    """

    def __init__(
        self,
        color,
        depth=ALPHABETA_DEFAULT_DEPTH,
        prunning=False,
        value_fn_builder_name=None,
        params=DEFAULT_WEIGHTS,
        epsilon=None,
    ):
        super().__init__(color)
        self.depth = int(depth)
        self.prunning = str(prunning).lower() != "false"
        self.value_fn_builder_name = (
            "relationship_aware_fn" if value_fn_builder_name == "R"
            else "strategic_fn" if value_fn_builder_name == "S"
            else "contender_fn" if value_fn_builder_name == "C" else "base_fn"
        )
        self.params = params
        self.use_value_function = None
        self.epsilon = epsilon
        self.social_memory = { 
            "relationships": {},
            "grudges": {}
           
        }
        self.last_processed_idx = 0

    def value_function(self, game, p0_color):
        raise NotImplementedError

    def get_actions(self, game):
        if self.prunning:
            return list_prunned_actions(game)
        return game.playable_actions
    
    def update_memory(self, game):
        history = game.state.action_records
        
        # Ensure all players in the game have an entry in memory
        for p in game.state.players:
            c = p.color
            if c not in self.social_memory["relationships"]:
                self.social_memory["relationships"][c] = 1.0
            if c not in self.social_memory["grudges"]:
                self.social_memory["grudges"][c] = 0

        while self.last_processed_idx < len(history):
            record = history[self.last_processed_idx]
            self.last_processed_idx += 1
            action = record.action

            if hasattr(action, 'action_type') and action.action_type == "MOVE_ROBBER":
                thief = action.color
                # Use .getattr safely because target_color might be None
                victim = getattr(action, 'target_color', None)
                
                # If I was the victim, reduce relationship and add grudge if it was a different player
                if victim == self.color and thief != self.color:
                    # Tank the relationship
                    old_rel = self.social_memory["relationships"].get(thief, 1.0)
                    self.social_memory["relationships"][thief] = max(0.2, old_rel - 0.02)

                    # Add a grudge
                    self.social_memory["grudges"][thief] = self.social_memory["grudges"].get(thief, 0) + 1
                # If I robbed the thief, reduce grudge if exists
                elif thief == self.color and victim in self.social_memory["grudges"]:
                    # Reduce grudge if retribution happens
                    self.social_memory["grudges"][victim] = max(0, self.social_memory["grudges"][victim] - 1)

                 # If it didn't robbe no one but it was a different player, reduce grudge if exists
                elif victim is None and thief != self.color:
                    if thief in self.social_memory["grudges"]:
                        self.social_memory["grudges"][thief] = 0
                
                # If thief robbed someone else, reduce grudge if exists
                elif victim != self.color and thief != self.color:
                    if thief in self.social_memory["grudges"]:
                        self.social_memory["grudges"][thief] = max(0, self.social_memory["grudges"][thief] - 1)   

            elif hasattr(action, 'action_type') and action.action_type == ActionType.ACCEPT_TRADE:
                if action.value:
                    trader = action.color
                    if trader != self.color:
                        # Increase relationship for successful trade
                        old_rel = self.social_memory["relationships"].get(trader, 1.0)
                        self.social_memory["relationships"][trader] = min(2.0, old_rel + 0.1)

            elif hasattr(action, 'action_type') and action.action_type == ActionType.REJECT_TRADE:
                # Unsuccessful trade attempt might indicate a strained relationship
                if not action.value:  # Assuming action.value indicates failure
                    trader = action.color
                    if trader != self.color:
                        old_rel = self.social_memory["relationships"].get(trader, 1.0)
                        self.social_memory["relationships"][trader] = max(0.2, old_rel - 0.01)

        for player in self.social_memory["grudges"]:
            if self.social_memory["grudges"][player] > 0:
                self.social_memory["grudges"][player] = max(0, self.social_memory["grudges"][player] - 0.1)
        for player in self.social_memory["relationships"]:
            if self.social_memory["relationships"][player] > 0.2:
                self.social_memory["relationships"][player] = min(2.0, self.social_memory["relationships"][player] + 0.05)


    def decide(self, game: Game, playable_actions):
        self.update_memory(game)

        actions = self.get_actions(game)
        if len(actions) == 1:
            return actions[0]

        if self.epsilon is not None and random.random() < self.epsilon:
            return random.choice(playable_actions)

        start = time.time()
        state_id = str(len(game.state.action_records))
        node = DebugStateNode(state_id, self.color)  # i think it comes from outside
        deadline = start + MAX_SEARCH_TIME_SECS
        result = self.alphabeta(
            game.copy(), self.depth, float("-inf"), float("inf"), deadline, node, self.social_memory
        )
        # print("Decision Results:", self.depth, len(actions), time.time() - start)
        # if game.state.num_turns > 10:
        #     render_debug_tree(node)
        #     breakpoint()
        if result[0] is None:
            return playable_actions[0]
        return result[0]

    def __repr__(self) -> str:
        return (
            super().__repr__()
            + f"(depth={self.depth},value_fn={self.value_fn_builder_name},prunning={self.prunning})"
        )

    def alphabeta(self, game, depth, alpha, beta, deadline, node, social_memory=None):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if depth == 0 or game.winning_color() is not None or time.time() >= deadline:
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
                social_memory
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        maximizingPlayer = game.state.current_color() == self.color
        actions = self.get_actions(game)  # list of actions.
        
        if depth < self.depth and self.value_fn_builder_name == "strategic_fn":
            actions = [
                a for a in actions 
                if not (hasattr(a, 'action_type') and a.action_type == ActionType.OFFER_TRADE)
            ]

        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        if maximizingPlayer:
            best_action = None
            best_value = float("-inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node, social_memory
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value > best_value:
                    best_action = action
                    best_value = expected_value
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # beta cutoff

            node.expected_value = best_value
            return best_action, best_value
        else:
            best_action = None
            best_value = float("inf")
            for i, (action, outcomes) in enumerate(action_outcomes.items()):
                action_node = DebugActionNode(action)

                expected_value = 0
                for j, (outcome, proba) in enumerate(outcomes):
                    out_node = DebugStateNode(
                        f"{node.label} {i} {j}", outcome.state.current_color()
                    )

                    result = self.alphabeta(
                        outcome, depth - 1, alpha, beta, deadline, out_node, social_memory
                    )
                    value = result[1]
                    expected_value += proba * value

                    action_node.children.append(out_node)
                    action_node.probas.append(proba)

                action_node.expected_value = expected_value
                node.children.append(action_node)

                if expected_value < best_value:
                    best_action = action
                    best_value = expected_value
                beta = min(beta, best_value)
                if beta <= alpha:
                    break  # alpha cutoff

            node.expected_value = best_value
            return best_action, best_value


class DebugStateNode:
    def __init__(self, label, color):
        self.label = label
        self.children = []  # DebugActionNode[]
        self.expected_value = None
        self.color = color


class DebugActionNode:
    def __init__(self, action):
        self.action = action
        self.expected_value: Any = None
        self.children = []  # DebugStateNode[]
        self.probas = []


# def render_debug_tree(node):
#     from graphviz import Digraph

#     dot = Digraph("AlphaBetaSearch")

#     agenda = [node]

#     while len(agenda) != 0:
#         tmp = agenda.pop()
#         dot.node(
#             tmp.label,
#             label=f"<{tmp.label}<br /><font point-size='10'>{tmp.expected_value}</font>>",
#             style="filled",
#             fillcolor=tmp.color.value,
#         )
#         for child in tmp.children:
#             action_label = (
#                 f"{tmp.label} - {str(child.action).replace('<', '').replace('>', '')}"
#             )
#             dot.node(
#                 action_label,
#                 label=f"<{action_label}<br /><font point-size='10'>{child.expected_value}</font>>",
#                 shape="box",
#             )
#             dot.edge(tmp.label, action_label)
#             for action_child, proba in zip(child.children, child.probas):
#                 dot.node(
#                     action_child.label,
#                     label=f"<{action_child.label}<br /><font point-size='10'>{action_child.expected_value}</font>>",
#                 )
#                 dot.edge(action_label, action_child.label, label=str(proba))
#                 agenda.append(action_child)
#     print(dot.render())


class SameTurnAlphaBetaPlayer(AlphaBetaPlayer):
    """
    Same like AlphaBeta but only within turn
    """

    def alphabeta(self, game, depth, alpha, beta, deadline, node, social_memory=None):
        """AlphaBeta MiniMax Algorithm.

        NOTE: Sometimes returns a value, sometimes an (action, value). This is
        because some levels are state=>action, some are action=>state and in
        action=>state would probably need (action, proba, value) as return type.

        {'value', 'action'|None if leaf, 'node' }
        """
        if (
            depth == 0
            or game.state.current_color() != self.color
            or game.winning_color() is not None
            or time.time() >= deadline
        ):
            value_fn = get_value_fn(
                self.value_fn_builder_name,
                self.params,
                self.value_function if self.use_value_function else None,
                social_memory
            )
            value = value_fn(game, self.color)

            node.expected_value = value
            return None, value

        actions = self.get_actions(game)  # list of actions.
        action_outcomes = expand_spectrum(game, actions)  # action => (game, proba)[]

        best_action = None
        best_value = float("-inf")
        for i, (action, outcomes) in enumerate(action_outcomes.items()):
            action_node = DebugActionNode(action)

            expected_value = 0
            for j, (outcome, proba) in enumerate(outcomes):
                out_node = DebugStateNode(
                    f"{node.label} {i} {j}", outcome.state.current_color()
                )

                result = self.alphabeta(
                    outcome, depth - 1, alpha, beta, deadline, out_node, social_memory
                )
                value = result[1]
                expected_value += proba * value

                action_node.children.append(out_node)
                action_node.probas.append(proba)

            action_node.expected_value = expected_value
            node.children.append(action_node)

            if expected_value > best_value:
                best_action = action
                best_value = expected_value
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break  # beta cutoff

        node.expected_value = best_value
        return best_action, best_value
