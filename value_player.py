from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from time import sleep
import pprint

class ValuePlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    equity = self.estimate_equity(hole_card, round_state)

    if equity >= 0.75:
      for i in valid_actions:
        if i["action"] == "raise":
          action = i["action"]
          return action  # action returned here is sent to the poker engine

    action = valid_actions[1]["action"]
    return action # action returned here is sent to the poker engine

  def estimate_equity(self, hole_card, round_state):
    hole_cards = gen_cards(hole_card)
    community_cards = gen_cards(round_state["community_card"])
    return estimate_hole_card_win_rate(100, 2, hole_cards, community_cards)

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
  return ValuePlayer()
