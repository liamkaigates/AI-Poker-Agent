from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from time import sleep
import pprint

class CustomPlayer(BasePokerPlayer):

  def __init__(self):
    super(CustomPlayer, self).__init__()
    self.opponent_stats = {}

  def declare_action(self, valid_actions, hole_card, round_state):
    actions = [action["action"] for action in valid_actions] # Set actions to a list of valid actions
    best_action = "fold" # Set minimum action to fold
    best_value = float("-inf") # Set minimum action value to negative infity

    for action in actions:
      value = self.expectiminimax(action, hole_card, round_state) # Compute expectiminimax value

      if value > best_value: # Update best action/value if current action has a higher value
        best_action = action
        best_value = value

    return best_action

  def expectiminimax(self, action, hole_card, round_state):
    hole_cards = gen_cards(hole_card) # Generate Card objects for hole cards
    community_cards = gen_cards(round_state["community_card"]) # Generate Card objects for community cards
    street = round_state["street"] # Obtain street from round_state
    equity = estimate_hole_card_win_rate(self.simulation_count(street), 2, hole_cards, community_cards) # Compute equity using Monte Carlo simulations
    pot_amount = round_state["pot"]["main"]["amount"] # Obtain pot amount from round_state
    player_bet, opponent_bet = self.compute_bets(round_state) # Compute the bets set by the player and opponent
    current_bet = max(player_bet, opponent_bet) # Max bet is the current bet of the round
    call_cost = max(0, current_bet - player_bet) # Compute the cost to call
    raise_cost = self.raise_cost(round_state, player_bet, current_bet) # Compute the cost to raise
    buckets = self.get_buckets(equity, pot_amount, hole_cards, community_cards, round_state) # Groups hands into buckets based on strategic similiarity

    if action == "fold":
      return self.get_fold_value(call_cost, player_bet)

    if action == "call":
      return self.get_call_value(equity, pot_amount, call_cost, buckets)

    if action == "raise":
      return self.get_raise_value(equity, pot_amount, raise_cost, buckets, street)

    return float("-inf")

  def get_buckets(self, equity, pot_amount, hole_cards, community_cards, round_state):
    behavior = self.get_opponent_strategy(round_state)
    return {
      "equity": self.get_equity_bucket(equity),
      "hand": self.get_hand_bucket(hole_cards, community_cards),
      "potential": self.get_potential_bucket(hole_cards, community_cards),
      "pot": self.get_pot_amount_bucket(pot_amount),
      "opponent": self.get_opponent_bucket(behavior),
      "pressure": self.get_pressure_bucket(round_state),
      "behavior": behavior
    }

  def simulation_count(self, street):
    if street == "preflop":
      return 80
    if street == "flop":
      return 110
    if street == "turn":
      return 130
    return 150

  def compute_bets(self, round_state):
    histories = round_state["action_histories"].get(round_state["street"], [])
    player_bet = 0
    opponent_bet = 0

    for action in histories:
      if "amount" not in action:
        continue

      amount = action["amount"]

      if action.get("uuid") == self.uuid:
        player_bet = max(player_bet, amount)
      else:
        opponent_bet = max(opponent_bet, amount)

    return player_bet, opponent_bet

  def get_street_bet_size(self, street):
    if street in ["preflop", "flop"]:
      return 20
    return 40

  def raise_cost(self, round_state, player_bet, current_bet):
    total_cost = current_bet + self.get_street_bet_size(round_state["street"])
    return max(0, total_cost - player_bet)

  def get_fold_value(self, call_cost, player_cost):
    if call_cost == 0:
      return -1000.0
    return -0.25 * player_cost

  def get_call_value(self, equity, pot_amount, call_cost, buckets):
    showdown_ev = equity * (pot_amount + call_cost) - call_cost

    if call_cost == 0:
      return showdown_ev * 1.25

    pot_amount_bonus = 3.0 * buckets["pot_amount"]
    potential_bonus = 5.0 * buckets["potential"]
    hand_bonus = 4.0 * buckets["hand"]
    pressure_penalty = 3.0 * buckets["pressure"]
    opponent_behavior_penalty = buckets["behavior"] * call_cost * 0.25

    return showdown_ev + pot_amount_bonus + potential_bonus + hand_bonus - pressure_penalty - opponent_behavior_penalty

  def _raise_value(self, equity, pot_amount, raise_cost, buckets, street):
    fold_probability = self.get_fold_probability(buckets, street)
    raise_probability = 1.0 - fold_probability
    called_pot = pot_amount + (2 * raise_cost)

    immediate_win_ev = fold_probability * pot_amount
    called_ev = continue_probability * (equity * called_pot - raise_cost)
    value_bonus = 6.0 * buckets["equity"] * buckets["pot"]
    made_hand_bonus = 8.0 * buckets["hand"]
    semi_bluff_bonus = 6.0 * buckets["potential"] * (1.0 - buckets["opponent"])
    pressure_penalty = buckets["pressure"] * raise_cost * 0.25

    return immediate_win_ev + called_ev + value_bonus + made_hand_bonus + semi_bluff_bonus - pressure_penalty

  def get_fold_probability(self, buckets, street):
    probability = 0.25
    probability += 0.04 * (1.0 - buckets["opponent"])
    probability += 0.05 * buckets["equity"]
    probability += 0.04 * buckets["hand"]
    probability += 0.03 * buckets["pressure"]

    if street == "preflop":
      probability += 0.04
    elif street == "river":
      probability -= 0.04

    return min(0.55, max(0.08, probability))

  def get_equity_bucket(self, equity):
    if equity < 0.35:
      return 0.0
    if equity < 0.55:
      return 0.5
    if equity < 0.75:
      return 0.8
    return 1.0

  def get_pot_bucket(self, pot):
    if pot < 80:
      return 0.0
    if pot < 180:
      return 0.5
    return 1.0

  def get_hand_bucket(self, hole_cards, community_cards):
    rank_info = HandEvaluator.gen_hand_rank_info(hole_cards, community_cards)
    strength = rank_info["hand"]["strength"]

    if strength == "HIGHCARD":
      return 0.0
    if strength == "ONEPAIR":
      return 0.25
    if strength == "TWOPAIR":
      return 0.45
    if strength == "THREECARD":
      return 0.65
    if strength == "STRAIGHT":
      return 0.75
    if strength == "FLASH":
      return 0.82
    if strength == "FULLHOUSE":
      return 0.90
    if strength == "FOURCARD":
      return 0.96
    return 1.0

  def get_potential_bucket(self, hole_cards, community_cards):
    cards = hole_cards + community_cards
    if len(community_cards) == 5:
      return 0.0

    flush_draw = self.get_flush_draw(cards)
    straight_draw = self.get_straight_draw(cards)

    if flush_draw and straight_draw:
      return 1.0
    if flush_draw or straight_draw:
      return 0.6
    if self.get_pair_or_better(cards):
      return 0.3
    return 0.0

  def get_flush_draw(self, cards):
    suit_counts = {}
    for card in cards:
      suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
    return max(suit_counts.values()) >= 4

  def get_straight_draw(self, cards):
    ranks = set([card.rank for card in cards])
    if 14 in ranks:
      ranks.add(1)

    for low_rank in range(1, 11):
      window = set(range(low_rank, low_rank + 5))
      if len(window.intersection(ranks)) >= 4:
        return True

    return False

  def get_pair_or_better(self, cards):
    rank_counts = {}
    for card in cards:
      rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
    return max(rank_counts.values()) >= 2

  def get_opponent_bucket(self, behavior):
    if behavior < 0.30:
      return 0.0
    if behavior < 0.60:
      return 0.5
    return 1.0

  def get_pressure_bucket(self, round_state):
    histories = round_state["action_histories"].get(round_state["street"], [])
    raises = 0

    for action in histories:
      if action_state["action"] == "RAISE":
        raises += 1

    if raises == 0:
      return 0.0
    if raises == 1:
      return 0.5
    return 1.0

  def get_opponent_strategy(self, round_state):
    raises = 0
    calls = 0
    folds = 0

    for histories in round_state["action_histories"].values():
      for action_state in histories:
        if action_state.get("uuid") == self.uuid:
          continue

        if action_state["action"] == "RAISE":
          raises += 1
        elif action_state["action"] == "CALL":
          calls += 1
        elif action_state["action"] == "FOLD":
          folds += 1

    for stats in self.opponent_stats.values():
      raises += stats["raises"]
      calls += stats["calls"]
      folds += stats["folds"]

    return (raises + 0.5) / (raises + calls + folds + 1.5)

  def receive_game_start_message(self, game_info):
    pass

  def receive_round_start_message(self, round_count, hole_card, seats):
    pass

  def receive_street_start_message(self, street, round_state):
    pass

  def receive_game_update_message(self, action, round_state):
    pass

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass


def setup_ai():
  return CustomPlayer()
