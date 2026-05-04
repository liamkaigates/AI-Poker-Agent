from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from time import sleep
import pprint

class CustomPlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    actions = [action["action"] for action in valid_actions] # Set actions to a list of valid actions
    max_action = "fold" # Set minimum action to fold
    max_value = float("-inf") # Set minimum action value to negative infity

    for action in actions:
      value = self.expectiminimax(action, hole_card, round_state) # Compute expectiminimax value

      if value > max_value: # Update best action/value if current action has a higher value
        max_value = value
        max_action = action

    return max_action # Return the best action

  def expectiminimax(self, action, hole_card, round_state):
    hole_cards = gen_cards(hole_card) # Generate Card objects for hole cards
    community_cards = gen_cards(round_state["community_card"]) # Generate Card objects for community cards
    street = round_state["street"] # Obtain round type (preflop, flop, turn, river, showdown)
    win_rate = estimate_hole_card_win_rate(250, 2, hole_cards, community_cards) # Compute win rate using Monte Carlo simulations
    pot_amount = round_state["pot"]["main"]["amount"] # Obtain pot amount
    action_histories = round_state["action_histories"].get(street, []) # Obtain action history of current round
    fold_probability = 0.2 # Assume the opponent folds with probaility of 0.2
    player_amount = 0.0
    max_amount = 0.0

    # Compute if a call is free and the player amount
    for action_state in action_histories:
      if "amount" not in action_state:
        continue

      amount = action_state["amount"]
      max_amount = max(max_amount, amount)

      if action_state.get("uuid") == self.uuid:
        player_amount = max(player_amount, amount)

    call_is_free = player_amount == max_amount # Check if calling is free

    if action == "fold":
      if call_is_free:
        return -1000.0 # The player should never call if it is free
      return -player_amount # Folding results in losing all current bets

    if action == "call":
      return win_rate * pot_amount - (1 - win_rate) * max_amount # Expected value of the action

    if action == "raise":
      return fold_probability * pot_amount - (1 - fold_probability) * (win_rate * pot_amount - (1 - win_rate) * max(1000, pot_amount + 10)) # Expected value of the action

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
