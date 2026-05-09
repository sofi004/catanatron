import math
from collections import defaultdict

from catanatron.game import Game
from catanatron.models.actions import generate_playable_actions
from catanatron.models.map import number_probability
from catanatron.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    SETTLEMENT,
    CITY,
    Action,
    ActionRecord,
    ActionType,
    ActionPrompt,
)
from catanatron.state_functions import (
    get_player_buildings,
    get_dev_cards_in_hand,
    get_player_freqdeck,
    get_enemy_colors,
)
from catanatron.features import build_production_features
from catanatron.players.value import value_production

DETERMINISTIC_ACTIONS = set(
    [
        ActionType.END_TURN,
        ActionType.BUILD_SETTLEMENT,
        ActionType.BUILD_ROAD,
        ActionType.BUILD_CITY,
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.PLAY_YEAR_OF_PLENTY,
        ActionType.PLAY_ROAD_BUILDING,
        ActionType.MARITIME_TRADE,
        ActionType.REJECT_TRADE,
        ActionType.CONFIRM_TRADE,
        ActionType.CANCEL_TRADE,
        ActionType.DISCARD_RESOURCE,  # for simplicity... ok if reality is slightly different
        ActionType.PLAY_MONOPOLY,  # for simplicity... we assume good card-counting and bank is visible...
    ]
)


def execute_deterministic(game, action):
    copy = game.copy()
    copy.execute(action, validate_action=False)
    return [(copy, 1)]


def execute_offer_trade_spectrum(game: Game, action: Action):
    """For targeted OFFER_TRADE (11-length), assume acceptance and directly simulate
    the completed trade state with probability 1.0. This avoids search getting stuck
    in negotiation states and lets minimax evaluate the actual trade outcome.
    """
    value = action.value
    # Check if this is a targeted trade (11-tuple format)
    if not isinstance(value, (list, tuple)) or len(value) < 11:
        # Fall back to deterministic behavior for non-targeted trades
        return execute_deterministic(game, action)
    
    # Extract trade components
    offered = value[:5]
    asked = value[5:10]
    target_color = value[10]
    
    # Resolve string target to Color enum if necessary
    if isinstance(target_color, str):
        resolved = None
        for c in game.state.colors:
            if c.name == target_color or c.name == target_color.upper():
                resolved = c
                break
        target_color = resolved
    
    if target_color is None or target_color == action.color:
        # Invalid target, fall back to deterministic
        return execute_deterministic(game, action)
    
    # Simulate the completed trade directly
    game_copy = game.copy()
    state = game_copy.state
    
    offerer_idx = state.color_to_index[action.color]
    target_idx = state.color_to_index[target_color]
    offerer_key = f"P{offerer_idx}"
    target_key = f"P{target_idx}"
    
    # Transfer resources
    for i, resource in enumerate(RESOURCES):
        # Offerer gives offered[i], receives asked[i]
        state.player_state[f"{offerer_key}_{resource}_IN_HAND"] -= offered[i]
        state.player_state[f"{offerer_key}_{resource}_IN_HAND"] += asked[i]
        
        # Target receives offered[i], gives asked[i]
        state.player_state[f"{target_key}_{resource}_IN_HAND"] += offered[i]
        state.player_state[f"{target_key}_{resource}_IN_HAND"] -= asked[i]
    
    # Transition state back to PLAY_TURN (as if trade completed)
    state.is_resolving_trade = False
    state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    state.acceptees = tuple(False for _ in state.colors)
    state.current_trade_targets = tuple(False for _ in state.colors)
    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    game_copy.playable_actions = generate_playable_actions(state)
    
    return [(game_copy, 1.0)]


def execute_accept_trade_spectrum(game: Game, action: Action):
    """For ACCEPT_TRADE, assume the trade eventually completes and directly simulate
    the final trade state with probability 1.0. If the accepting player is evaluating
    their own trade acceptance, they should see the actual benefit of accepting.
    """
    state = game.state
    
    # Extract current trade components
    offered = state.current_trade[:5]
    asked = state.current_trade[5:10]
    offerer_color = state.colors[state.current_turn_index]
    accepter_color = action.color
    
    if offerer_color is None or accepter_color is None:
        # Invalid state, fall back to deterministic
        return execute_deterministic(game, action)
    
    # Simulate the completed trade directly
    game_copy = game.copy()
    state_copy = game_copy.state
    
    offerer_idx = state_copy.color_to_index[offerer_color]
    accepter_idx = state_copy.color_to_index[accepter_color]
    offerer_key = f"P{offerer_idx}"
    accepter_key = f"P{accepter_idx}"
    
    # Transfer resources
    for i, resource in enumerate(RESOURCES):
        # Offerer gives offered[i], receives asked[i]
        state_copy.player_state[f"{offerer_key}_{resource}_IN_HAND"] -= offered[i]
        state_copy.player_state[f"{offerer_key}_{resource}_IN_HAND"] += asked[i]
        
        # Accepter receives offered[i], gives asked[i]
        state_copy.player_state[f"{accepter_key}_{resource}_IN_HAND"] += offered[i]
        state_copy.player_state[f"{accepter_key}_{resource}_IN_HAND"] -= asked[i]
    
    # Transition state back to PLAY_TURN (as if trade completed)
    state_copy.is_resolving_trade = False
    state_copy.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    state_copy.acceptees = tuple(False for _ in state_copy.colors)
    state_copy.current_trade_targets = tuple(False for _ in state_copy.colors)
    state_copy.current_player_index = state_copy.current_turn_index
    state_copy.current_prompt = ActionPrompt.PLAY_TURN
    game_copy.playable_actions = generate_playable_actions(state_copy)
    
    return [(game_copy, 1.0)]


def execute_spectrum(game: Game, action: Action):
    """Returns [(game_copy, proba), ...] tuples for result of given action.
    Result probas should add up to 1. Does not modify self"""
    if action.action_type == ActionType.OFFER_TRADE:
        return execute_offer_trade_spectrum(game, action)
    elif action.action_type == ActionType.ACCEPT_TRADE:
        return execute_accept_trade_spectrum(game, action)
    elif action.action_type in DETERMINISTIC_ACTIONS:
        return execute_deterministic(game, action)
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        results = []

        # Get the possible deck from the perspective of the current player
        # by getting all face down cards
        current_deck = game.state.development_listdeck.copy()
        for color in get_enemy_colors(game.state.colors, action.color):
            for card in DEVELOPMENT_CARDS:
                number = get_dev_cards_in_hand(game.state, color, card)
                current_deck += [card] * number

        for card in set(current_deck):
            option_action = Action(action.color, action.action_type, card)
            option_game = game.copy()
            try:
                option_game.execute(option_action, validate_action=False)
            except Exception:
                # ignore exceptions, since player might imagine impossible outcomes.
                # ignoring means the value function of this node will be flattened,
                # to the one before.
                pass
            results.append((option_game, current_deck.count(card) / len(current_deck)))
        return results
    elif action.action_type == ActionType.ROLL:
        results = []
        for roll in range(2, 13):
            outcome = (roll // 2, math.ceil(roll / 2))

            option_action = Action(action.color, action.action_type, outcome)
            option_game = game.copy()
            option_game.execute(option_action, validate_action=False)
            results.append((option_game, number_probability(roll)))
        return results
    elif action.action_type == ActionType.MOVE_ROBBER:
        (coordinate, robbed_color) = action.value
        if robbed_color is None:  # no one to steal, then deterministic
            return execute_deterministic(game, action)

        results = []
        opponent_hand = get_player_freqdeck(game.state, robbed_color)
        opponent_hand_size = sum(opponent_hand)
        if opponent_hand_size == 0:
            # Nothing to steal
            return execute_deterministic(game, action)

        for card in RESOURCES:
            option_action = Action(
                action.color,
                action.action_type,
                (coordinate, robbed_color),
            )
            option_action_record = ActionRecord(action=option_action, result=card)
            option_game = game.copy()
            try:
                option_game.execute(
                    option_action,
                    validate_action=False,
                    action_record=option_action_record,
                )
            except Exception:
                # ignore exceptions, since player might imagine impossible outcomes.
                # ignoring means the value function of this node will be flattened,
                # to the one before.
                pass
            results.append((option_game, 1 / 5.0))
        return results
    else:
        raise RuntimeError("Unknown ActionType " + str(action.action_type))


def expand_spectrum(game, actions):
    """Consumes game if playable_actions not specified"""
    children = defaultdict(list)
    for action in actions:
        outprobas = execute_spectrum(game, action)
        children[action] = outprobas
    return children  # action => (game, proba)[]


def list_prunned_actions(game: Game):
    current_color = game.state.current_color()
    playable_actions = game.playable_actions
    actions = playable_actions.copy()
    types = set(map(lambda a: a.action_type, playable_actions))

    # Prune Initial Settlements at 1-tile places
    if ActionType.BUILD_SETTLEMENT in types and game.state.is_initial_build_phase:
        actions = filter(
            lambda a: len(game.state.board.map.adjacent_tiles[a.value]) != 1, actions
        )

    # Prune Trading if can hold for resources. Only for rare resources.
    if ActionType.MARITIME_TRADE in types:
        port_resources = game.state.board.get_player_port_resources(current_color)
        has_three_to_one = None in port_resources
        # TODO: for 2:1 ports, skip any 3:1 or 4:1 trades
        # TODO: if can_safely_hold, prune all
        tmp_actions = []
        for action in actions:
            if action.action_type != ActionType.MARITIME_TRADE:
                tmp_actions.append(action)
                continue
            # has 3:1, skip any 4:1 trades
            if has_three_to_one and action.value[3] is not None:
                continue
            tmp_actions.append(action)
        actions = tmp_actions

    if ActionType.MOVE_ROBBER in types:
        actions = prune_robber_actions(current_color, game, actions)

    return list(actions)


def prune_robber_actions(current_color, game, actions):
    """Eliminate all but the most impactful tile"""
    enemy_color = next(filter(lambda c: c != current_color, game.state.colors))
    enemy_owned_tiles = set()
    for node_id in get_player_buildings(game.state, enemy_color, SETTLEMENT):
        enemy_owned_tiles.update(game.state.board.map.adjacent_tiles[node_id])
    for node_id in get_player_buildings(game.state, enemy_color, CITY):
        enemy_owned_tiles.update(game.state.board.map.adjacent_tiles[node_id])

    robber_moves = set(
        filter(
            lambda a: a.action_type == ActionType.MOVE_ROBBER
            and game.state.board.map.tiles[a.value[0]] in enemy_owned_tiles,
            actions,
        )
    )

    if len(robber_moves) == 0:
        return actions

    production_features = build_production_features(True)

    def impact(action):
        game_copy = game.copy()
        game_copy.execute(action)

        our_production_sample = production_features(game_copy, current_color)
        enemy_production_sample = production_features(game_copy, current_color)
        production = value_production(our_production_sample, "P0")
        enemy_production = value_production(enemy_production_sample, "P1")

        return enemy_production - production

    most_impactful_robber_action = max(
        robber_moves, key=impact
    )  # most production and variety producing
    actions = filter(
        lambda a: a.action_type != ActionType.MOVE_ROBBER
        or a == most_impactful_robber_action,
        # lambda a: a.action_type != ActionType.MOVE_ROBBER or a in robber_moves,
        actions,
    )
    return actions
