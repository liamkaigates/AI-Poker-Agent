from pypokerengine.api.game import setup_config, start_poker
from call_player import CallPlayer
from equity_player import EquityPlayer
from fold_player import FoldPlayer
from randomplayer import RandomPlayer
from raise_player import RaisedPlayer
from submission.custom_player import CustomPlayer
from value_player import ValuePlayer

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=10000, small_blind_amount=10)


config.register_player(name="f1", algorithm=RandomPlayer())
# config.register_player(name="FT2", algorithm=RaisedPlayer())
# config.register_player(name="FT2", algorithm=CallPlayer())
# config.register_player(name="FT2", algorithm=ValuePlayer())
# config.register_player(name="FT2", algorithm=EquityPlayer())
# config.register_player(name="FT2", algorithm=FoldPlayer())
config.register_player(name="f2", algorithm=CustomPlayer())

game_result = start_poker(config, verbose=1)
