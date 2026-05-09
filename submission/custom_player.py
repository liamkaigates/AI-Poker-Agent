from pypokerengine.players import BasePokerPlayer
from pypokerengine.engine.hand_evaluator import HandEvaluator
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

MAX_DEPTH = 2 # Max depth for expectiminimax tree search
ACTION_LIKELIHOODS = { # We define 5 levels of hand strength and the likelihood of each action given the respective hand strength
  "fold": [0.60, 0.35, 0.20, 0.08, 0.03],
  "call": [0.30, 0.45, 0.50, 0.37, 0.27],
  "raise": [0.10, 0.20, 0.30, 0.55, 0.70],
}


# Use class to keep track of weights used in agent and training 
class EvaluationWeights:
  ATTRIBUTES = ("equity", "hand", "potential", "pot", "opponent", "opponent_equity", "pressure", "behavior", "cost", "risk") # Attributes used by each weight

  # Set weights for the agent
  def __init__(self, equity=2.48, hand=7.82, potential=5.10, pot=0.84, opponent=0.14, opponent_equity=0.88, pressure=4.15, behavior=0.10, cost=0.03, risk=0.27):
    self.equity = equity
    self.hand = hand
    self.potential = potential
    self.pot = pot
    self.opponent = opponent
    self.opponent_equity = opponent_equity
    self.pressure = pressure
    self.behavior = behavior
    self.cost = cost
    self.risk = risk

  # 
  def copy(self, **updates):
    values = {attr: getattr(self, attr) for attr in self.ATTRIBUTES}
    values.update(updates)
    return EvaluationWeights(**values)


class SearchConfig:
  def __init__(self, max_depth=MAX_DEPTH, random_seed=None, weights=None):
    self.max_depth = max_depth
    self.random_seed = random_seed
    self.weights = weights or EvaluationWeights()


class OpponentModel:
  # A class that stores observed information about the opponent to predict their hand strength and actions
  def __init__(self):
    self.beliefs = [0.2] * 5 # Initialized to assume the opponent is equally likely to have any hand strength at the beginning of the game
    self.raise_count = 0.0
    self.call_count = 0.0
    self.fold_count = 0.0
    self.total = 0.0
    self.late_raise_pressure = 0.0 # This value increases in later streets, to indicate that a raise has higher signifcance when the pot and information is more developed.
    self.street_totals = {
      "preflop": 0.0,
      "flop": 0.0,
      "turn": 0.0,
      "river": 0.0,
    }

  def observe(self, action, street=None):
    # Observe an action from the opponent and update beliefs and action counts accordingly. Actions in later streets are weighted more heavily.
    action = action.lower()
    if action not in ACTION_LIKELIHOODS:
      return

    # Get weight for given street and update action count
    weight = self.street_weight(street)
    if action == "raise":
      self.raise_count += weight
      if street in ("turn", "river"):
        self.late_raise_pressure += weight
    elif action == "call":
      self.call_count += weight
    elif action == "fold":
      self.fold_count += weight
    self.total += weight

    if street in self.street_totals:
      self.street_totals[street] += weight

    # Update beliefs about opponent's hand strength based on the observed action, normalizing it to a probability distribution.
    likelihoods = ACTION_LIKELIHOODS[action]
    new_beliefs = [
      belief * (likelihood ** weight)
      for belief, likelihood in zip(self.beliefs, likelihoods)
    ]
    normalizer = sum(new_beliefs)
    if normalizer > 0:
      self.beliefs = [belief / normalizer for belief in new_beliefs]

  def street_weight(self, street):
     # Weight defining how informative actions are in different game phases.
    if street == "preflop":
      return 0.60
    if street == "flop":
      return 0.90
    if street == "turn":
      return 1.20
    if street == "river":
      return 1.50
    return 1.0

  def fold_probability(self):
    # Current estimate of the probaility the opponent will fold (initially 30%)
    if self.total == 0:
      return 0.30
    passive_rate = (self.fold_count + (0.4 * self.call_count)) / self.total
    weight = min(self.total / 25.0, 1.0)
    return (weight * passive_rate) + ((1.0 - weight) * 0.30)

  def call_probability(self):
    # Current estimate of the probaility the opponent will call (initially 40%)
    if self.total == 0:
      return 0.40
    rate = self.call_count / self.total
    weight = min(self.total / 25.0, 1.0)
    return (weight * rate) + ((1.0 - weight) * 0.40)

  def aggression_factor(self):
    # A measure of how aggresive the opponent's actions are
    if self.total == 0:
      return 0.5
    return (self.raise_count + (0.25 * self.call_count)) / self.total

  def estimated_equity(self):
    # A rough estimate of the opponent's expected strength based on the current observations and game state
    expected_strength = sum(index * probability for index, probability in enumerate(self.beliefs))
    late_pressure_bonus = min(0.25, 0.08 * self.late_raise_pressure)
    return min(1.0, (expected_strength / 4.0) + late_pressure_bonus)


class CustomPlayer(BasePokerPlayer):

  def __init__(self):
    super(CustomPlayer, self).__init__()
    self.opponent_model = OpponentModel()
    self.search_config = SearchConfig()
    self.search_cache = {}

  def declare_action(self, valid_actions, hole_card, round_state):
    actions = [
      action["action"]
      for action in valid_actions
      if action["action"] != "raise" or self.is_raise_legal(action)
    ]
    self.search_cache = {}
    context = self.build_context(hole_card, round_state, valid_actions)
    best_action = "fold"
    best_value = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for action in actions:
      value = self.expectiminimax(action, context, self.search_config.max_depth, alpha, beta)
      if value > best_value:
        best_action = action
        best_value = value
      alpha = max(alpha, best_value)

    return best_action

  def expectiminimax(self, action, context, depth, alpha=float("-inf"), beta=float("inf")):
    cache_key = self.search_cache_key(action, context, depth)
    if cache_key in self.search_cache:
      return self.search_cache[cache_key]

    if action == "fold":
      value = self.get_fold_value(context["call_cost"], context["player_bet"])
      self.search_cache[cache_key] = value
      return value

    if action == "call":
      value = self.get_call_value(context, depth, alpha, beta)
      self.search_cache[cache_key] = value
      return value

    if action == "raise":
      if context["raises_left"] <= 0:
        value = self.get_call_value(context, depth, alpha, beta)
      else:
        value = self.get_raise_value(context, depth, alpha, beta)
      self.search_cache[cache_key] = value
      return value

    return float("-inf")

  def search_cache_key(self, action, context, depth):
    buckets = context["buckets"]
    return (
      action,
      depth,
      context["street"],
      round(context["equity"], 2),
      self.get_pot_amount_bucket(context["pot_amount"]),
      min(context["call_cost"], 120),
      min(context["raise_cost"], 120),
      context["raises_left"],
      buckets["equity"],
      buckets["hand"],
      buckets["potential"],
      buckets["pot"],
      buckets["opponent"],
      round(buckets["opponent_equity"], 2),
      buckets["pressure"],
      round(buckets["behavior"], 2),
    )

  def get_call_value(self, context, depth, alpha=float("-inf"), beta=float("inf")):
    equity = context["equity"]
    pot_amount = context["pot_amount"]
    call_cost = context["call_cost"]
    buckets = context["buckets"]

    current_ev = self.leaf_value(equity, pot_amount + call_cost, call_cost, street=context["street"])
    if call_cost == 0:
      current_ev *= 1.25
    else:
      pot_odds = call_cost / max(1.0, pot_amount + call_cost)
      if equity > pot_odds + 0.08:
        current_ev += (equity - pot_odds) * min(pot_amount, 160) * 0.25
      if buckets["pressure"] >= 0.5 and buckets["opponent_equity"] >= 0.55:
        continue_threshold = max(pot_odds + 0.18, 0.55)
        if equity < continue_threshold:
          current_ev -= (continue_threshold - equity) * call_cost * 5.0

    if depth <= 1 or self.next_street(context["street"]) is None:
      return self.apply_bucket_adjustments(current_ev, call_cost, buckets, is_raise=False)

    future_context = self.advance_context(context, pot_amount + call_cost, 0)
    future_ev = self.best_future_choice(future_context, depth - 1, alpha, beta)
    weight = 1.0 if context["street"] == "river" else 0.55
    value = (weight * current_ev) + ((1.0 - weight) * future_ev)
    return self.apply_bucket_adjustments(value, call_cost, buckets, is_raise=False)

  def get_raise_value(self, context, depth, alpha=float("-inf"), beta=float("inf")):
    equity = context["equity"]
    pot_amount = context["pot_amount"]
    raise_cost = context["raise_cost"]
    street = context["street"]
    buckets = context["buckets"]
    opponent_model = context["opponent_model"]

    raw_fold_probability = opponent_model.fold_probability()
    raw_call_probability = opponent_model.call_probability()
    is_calling_station = raw_fold_probability < 0.22 or raw_call_probability > 0.50
    is_value_raise = equity >= self.value_raise_threshold(street)
    is_bluff_raise = equity < 0.50 and buckets["potential"] >= 0.6

    fold_probability = self.clamp(raw_fold_probability, 0.04, 0.52)
    call_probability = self.clamp(raw_call_probability, 0.12, 0.82)
    if is_calling_station:
      fold_probability *= 0.35
      call_probability = max(call_probability, 0.65)
    if not is_value_raise and not is_bluff_raise:
      fold_probability *= 0.55

    raise_probability = max(0.0, 1.0 - fold_probability - call_probability)
    normalizer = fold_probability + call_probability + raise_probability
    fold_probability /= normalizer
    call_probability /= normalizer
    raise_probability /= normalizer

    immediate_win_ev = pot_amount
    opponent_call_cost = self.get_street_bet_size(street)
    called_pot = pot_amount + raise_cost + opponent_call_cost
    called_ev = self.leaf_value(equity, called_pot, raise_cost, street=street)

    if depth > 1 and self.next_street(street) is not None:
      future_context = self.advance_context(context, called_pot, 0)
      future_ev = self.best_future_choice(future_context, depth - 1, alpha, beta)
      called_ev = (0.55 * called_ev) + (0.45 * future_ev)

    if depth <= 1 or context["raises_left"] <= 1:
      reraise_ev = max(
        self.get_fold_value(self.get_street_bet_size(street), raise_cost),
        self.leaf_value(
          equity,
          called_pot + self.get_street_bet_size(street),
          raise_cost + self.get_street_bet_size(street),
          street=street
        )
      )
    else:
      reraise_context = dict(context)
      reraise_context["pot_amount"] = called_pot + self.get_street_bet_size(street)
      reraise_context["call_cost"] = self.get_street_bet_size(street)
      reraise_context["raises_left"] = context["raises_left"] - 1
      reraise_ev = self.best_future_choice(reraise_context, depth - 1, alpha, beta)

    value = (
      fold_probability * immediate_win_ev
      + call_probability * called_ev
      + raise_probability * reraise_ev
    )
    value = self.apply_raise_risk_adjustment(
      value,
      equity,
      raise_cost,
      street,
      buckets,
      is_calling_station,
      is_value_raise,
      is_bluff_raise
    )
    return self.apply_bucket_adjustments(value, raise_cost, buckets, is_raise=True)

  def best_future_choice(self, context, depth, alpha=float("-inf"), beta=float("inf")):
    delta = self.equity_rate_change(context["street"])
    future_equities = [
      self.clamp(context["equity"] - delta, 0.0, 1.0),
      context["equity"],
      self.clamp(context["equity"] + delta, 0.0, 1.0),
    ]

    best = float("-inf")
    actions = ("raise", "call", "fold") if context["raises_left"] > 0 else ("call", "fold")
    for action in actions:
      total = 0.0
      for equity in future_equities:
        future_context = dict(context)
        future_context["equity"] = equity
        future_context["buckets"] = self.get_buckets(
          equity,
          future_context["pot_amount"],
          future_context["hole_cards"],
          future_context["community_cards"],
          future_context["round_state"],
          future_context["opponent_model"].aggression_factor(),
          future_context["opponent_model"].estimated_equity()
        )
        total += self.expectiminimax(action, future_context, depth, alpha, beta)
      action_value = total / len(future_equities)
      best = max(best, action_value)
      alpha = max(alpha, best)
      if alpha >= beta:
        break

    return best

  def build_context(self, hole_card, round_state, valid_actions):
    hole_cards = gen_cards(hole_card)
    community_cards = gen_cards(round_state["community_card"])
    street = round_state["street"]
    equity = estimate_hole_card_win_rate(
      self.simulation_count(street),
      2,
      hole_cards,
      community_cards
    )
    pot_amount = round_state["pot"]["main"]["amount"]
    player_bet, opponent_bet = self.compute_bets(round_state)
    current_bet = max(player_bet, opponent_bet)
    call_cost = self.get_call_cost(valid_actions, current_bet, player_bet)
    raise_cost = self.get_raise_cost(valid_actions, round_state, player_bet, current_bet)
    opponent_model = self.build_opponent_model(round_state)
    behavior = opponent_model.aggression_factor()

    return {
      "hole_cards": hole_cards,
      "community_cards": community_cards,
      "round_state": round_state,
      "street": street,
      "equity": equity,
      "pot_amount": pot_amount,
      "player_bet": player_bet,
      "opponent_bet": opponent_bet,
      "call_cost": call_cost,
      "raise_cost": raise_cost,
      "raises_left": self.raises_left(round_state),
      "opponent_model": opponent_model,
      "buckets": self.get_buckets(
        equity,
        pot_amount,
        hole_cards,
        community_cards,
        round_state,
        behavior,
        opponent_model.estimated_equity()
      ),
    }

  def advance_context(self, context, pot_amount, call_cost):
    next_context = dict(context)
    next_context["street"] = self.next_street(context["street"]) or context["street"]
    next_context["pot_amount"] = pot_amount
    next_context["call_cost"] = call_cost
    next_context["raise_cost"] = self.get_street_bet_size(next_context["street"])
    next_context["raises_left"] = 4
    return next_context

  def leaf_value(self, equity, pot_amount, contribution, street=None):
    weights = self.search_config.weights
    base_ev = (equity * pot_amount) - contribution

    # Risk aversion: avoid marginal negative swing spots
    risk_penalty = (1.0 - equity) * contribution * weights.risk

    # Pot-odds penalty when equity is below break-even
    breakeven = contribution / max(1.0, pot_amount + contribution)
    pot_odds_penalty = 0.0
    if equity < breakeven:
        pot_odds_penalty = (breakeven - equity) * contribution * 1.2

    # Street confidence
    confidence = {
        "preflop": 0.88,
        "flop": 0.93,
        "turn": 0.97,
        "river": 1.0,
    }.get(street, 1.0)

    return (base_ev - risk_penalty - pot_odds_penalty) * confidence

  def apply_bucket_adjustments(self, value, cost, buckets, is_raise):
    value += self.bucket_linear_value(buckets, cost)
    if is_raise:
      value += buckets["potential"] * (1.0 - buckets["opponent"]) * 0.45
      value -= buckets["pressure"] * cost * 0.05
    return value

  def apply_raise_risk_adjustment(self, value, equity, raise_cost, street, buckets, is_calling_station, is_value_raise, is_bluff_raise):
    if raise_cost <= 0:
      return value

    if is_calling_station and not is_value_raise:
      value -= raise_cost * 1.15
      value -= (0.55 - min(equity, 0.55)) * raise_cost * 2.0

    if not is_value_raise and not is_bluff_raise:
      value -= raise_cost * 0.45

    if buckets["pressure"] >= 0.5 and equity < self.value_raise_threshold(street) + 0.04:
      value -= raise_cost * buckets["pressure"] * 0.55

    if buckets["opponent_equity"] >= 0.60 and equity < self.value_raise_threshold(street) + 0.10:
      value -= raise_cost * buckets["opponent_equity"] * 1.10

    if is_value_raise:
      value += (equity - self.value_raise_threshold(street)) * raise_cost * 1.4

    return value

  def value_raise_threshold(self, street):
    if street == "preflop":
      return 0.62
    if street == "flop":
      return 0.58
    if street == "turn":
      return 0.56
    return 0.54

  def bucket_linear_value(self, buckets, cost):
    weights = self.search_config.weights
    return (
      weights.equity * buckets["equity"]
      + weights.hand * buckets["hand"]
      + weights.potential * buckets["potential"]
      + weights.pot * buckets["pot"]
      - weights.opponent * buckets["opponent"]
      - weights.opponent_equity * buckets["opponent_equity"]
      - weights.pressure * buckets["pressure"]
      - weights.behavior * buckets["behavior"] * cost * 0.25
      - weights.cost * cost
    )

  def get_fold_value(self, call_cost, player_cost):
    if call_cost == 0:
      return -1000.0
    return -0.25 * player_cost

  def get_buckets(self, equity, pot_amount, hole_cards, community_cards, round_state, behavior=None, opponent_equity=None):
    if behavior is None:
      behavior = self.get_opponent_strategy(round_state)
    if opponent_equity is None:
      opponent_equity = self.build_opponent_model(round_state).estimated_equity()
    return {
      "equity": self.get_equity_bucket(equity),
      "hand": self.get_hand_bucket(hole_cards, community_cards),
      "potential": self.get_potential_bucket(hole_cards, community_cards),
      "pot": self.get_pot_amount_bucket(pot_amount),
      "opponent": self.get_opponent_bucket(behavior),
      "opponent_equity": opponent_equity,
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
    if street in ("preflop", "flop"):
      return 20
    return 40

  def raise_cost(self, round_state, player_bet, current_bet):
    total_cost = current_bet + self.get_street_bet_size(round_state["street"])
    return max(0, total_cost - player_bet)

  def get_call_cost(self, valid_actions, current_bet, player_bet):
    for action in valid_actions:
      if action["action"] == "call":
        return action.get("amount", max(0, current_bet - player_bet))
    return max(0, current_bet - player_bet)

  def get_raise_cost(self, valid_actions, round_state, player_bet, current_bet):
    for action in valid_actions:
      if action["action"] == "raise" and self.is_raise_legal(action):
        amount = action.get("amount", {})
        if isinstance(amount, dict) and "min" in amount:
          return max(0, amount["min"] - player_bet)
        return self.raise_cost(round_state, player_bet, current_bet)
    return 0

  def is_raise_legal(self, action):
    amount = action.get("amount")
    if amount is None:
      return True
    if not isinstance(amount, dict):
      return False
    minimum = amount.get("min")
    maximum = amount.get("max")
    if minimum in (-1, None) or maximum in (-1, None):
      return False
    return minimum <= maximum

  def raises_left(self, round_state):
    histories = round_state["action_histories"].get(round_state["street"], [])
    raises = sum(1 for action in histories if action["action"] == "RAISE")
    return max(0, 4 - raises)

  def next_street(self, street):
    if street == "preflop":
      return "flop"
    if street == "flop":
      return "turn"
    if street == "turn":
      return "river"
    return None

  def equity_rate_change(self, street):
    if street == "preflop":
      return 0.20
    if street == "flop":
      return 0.12
    if street == "turn":
      return 0.07
    return 0.0

  def build_opponent_model(self, round_state):
    if self.opponent_model.total > 0:
      return self.opponent_model

    model = OpponentModel()
    for street, histories in round_state["action_histories"].items():
      for action in histories:
        if action.get("uuid") != self.uuid:
          model.observe(action["action"], street)
    return model

  def get_equity_bucket(self, equity):
    if equity < 0.35:
      return 0.0
    if equity < 0.55:
      return 0.5
    if equity < 0.75:
      return 0.8
    return 1.0

  def get_pot_amount_bucket(self, pot):
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
    raises = sum(1 for action in histories if action["action"] == "RAISE")

    if raises == 0:
      return 0.0
    if raises == 1:
      return 0.5
    return 1.0

  def get_opponent_strategy(self, round_state):
    model = self.build_opponent_model(round_state)
    return model.aggression_factor()

  def clamp(self, value, low, high):
    return min(high, max(low, value))

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
