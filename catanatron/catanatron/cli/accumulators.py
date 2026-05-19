import time
import os
import json
from collections import defaultdict

from catanatron.game import GameAccumulator, Game
from catanatron.json import GameEncoder
from catanatron.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_largest_army,
    get_longest_road_color,
    get_player_buildings,
)
from catanatron.models.enums import VICTORY_POINT, SETTLEMENT, CITY
from catanatron.models.actions import ActionType


class VpDistributionAccumulator(GameAccumulator):
    """
    Accumulates CITIES,SETTLEMENTS,DEVVPS,LONGEST,LARGEST
    in each game per player.
    """

    def __init__(self):
        # These are all per-player. e.g. self.cities['RED']
        self.cities = defaultdict(int)
        self.settlements = defaultdict(int)
        self.devvps = defaultdict(int)
        self.longest = defaultdict(int)
        self.largest = defaultdict(int)
        self.negotiation_attempts = defaultdict(int)
        self.negotiation_accepted = defaultdict(int)
        self.negotiation_rejected = defaultdict(int)

        self.num_games = 0

    def after(self, game: Game):
        winner = game.winning_color()
        if winner is None:
            return  # throw away data

        for color in game.state.colors:
            cities = len(get_player_buildings(game.state, color, CITY))
            settlements = len(get_player_buildings(game.state, color, SETTLEMENT))
            longest = get_longest_road_color(game.state) == color
            largest = get_largest_army(game.state)[0] == color
            devvps = get_dev_cards_in_hand(game.state, color, VICTORY_POINT)

            self.cities[color] += cities
            self.settlements[color] += settlements
            self.longest[color] += longest
            self.largest[color] += largest
            self.devvps[color] += devvps

        for action_record in game.state.action_records:
            action = action_record.action
            if action.action_type == ActionType.OFFER_TRADE:
                self.negotiation_attempts[action.color] += 1
            elif action.action_type == ActionType.CONFIRM_TRADE:
                self.negotiation_accepted[action.color] += 1
            elif action.action_type == ActionType.REJECT_TRADE:
                self.negotiation_rejected[action.color] += 1

        self.num_games += 1

    def get_avg_cities(self, color=None):
        if color is None:
            return sum(self.cities.values()) / self.num_games
        else:
            return self.cities[color] / self.num_games

    def get_avg_settlements(self, color=None):
        if color is None:
            return sum(self.settlements.values()) / self.num_games
        else:
            return self.settlements[color] / self.num_games

    def get_avg_longest(self, color=None):
        if color is None:
            return sum(self.longest.values()) / self.num_games
        else:
            return self.longest[color] / self.num_games

    def get_avg_largest(self, color=None):
        if color is None:
            return sum(self.largest.values()) / self.num_games
        else:
            return self.largest[color] / self.num_games

    def get_avg_devvps(self, color=None):
        if color is None:
            return sum(self.devvps.values()) / self.num_games
        else:
            return self.devvps[color] / self.num_games

    def get_avg_negotiation_attempts(self, color=None):
        if color is None:
            return sum(self.negotiation_attempts.values()) / self.num_games
        else:
            return self.negotiation_attempts[color] / self.num_games

    def get_avg_negotiation_accepted(self, color=None):
        if color is None:
            return sum(self.negotiation_accepted.values()) / self.num_games
        else:
            return self.negotiation_accepted[color] / self.num_games

    def get_avg_negotiation_rejected(self, color=None):
        if color is None:
            return sum(self.negotiation_rejected.values()) / self.num_games
        else:
            return self.negotiation_rejected[color] / self.num_games


class StatisticsAccumulator(GameAccumulator):
    def __init__(self):
        self.wins = defaultdict(int)
        self.turns = []
        self.ticks = []
        self.durations = []
        self.games = []
        self.results_by_player = defaultdict(list)
        self.max_turn_time_by_bot = defaultdict(float)
        self.bot_colors = set()

    def before(self, game):
        self.start = time.time()
        for player in game.state.players:
            if player.is_bot:
                self.bot_colors.add(player.color)

    def step(self, game_before_action, action):
        if getattr(game_before_action, "last_decision_was_auto", False):
            return

        color = getattr(game_before_action, "last_decision_color", None)
        duration = getattr(game_before_action, "last_decision_duration", None)
        if color is None or duration is None:
            return
        if color not in self.bot_colors:
            return

        self.max_turn_time_by_bot[color] = max(
            self.max_turn_time_by_bot[color], duration
        )

    def after(self, game):
        duration = time.time() - self.start
        winning_color = game.winning_color()
        if winning_color is None:
            return  # do not track

        self.wins[winning_color] += 1
        self.turns.append(game.state.num_turns)
        self.ticks.append(len(game.state.action_records))
        self.durations.append(duration)
        self.games.append(game)

        for color in game.state.colors:
            points = get_actual_victory_points(game.state, color)
            self.results_by_player[color].append(points)

    def get_avg_ticks(self):
        return sum(self.ticks) / len(self.ticks)

    def get_avg_turns(self):
        return sum(self.turns) / len(self.turns)

    def get_avg_duration(self):
        return sum(self.durations) / len(self.durations)

    def get_max_turn_time(self, color):
        return self.max_turn_time_by_bot[color]


class JsonDataAccumulator(GameAccumulator):
    def __init__(self, output):
        self.output = output

    def after(self, game):
        filepath = os.path.join(self.output, f"{game.id}.json")
        with open(filepath, "w") as f:
            f.write(json.dumps(game, cls=GameEncoder))
