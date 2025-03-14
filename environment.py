import numpy as np


class SkyjoEnv:
    def __init__(self, n_players=4, n_cards=150):
        self.grid_size = (3, 4)
        self.n_players = n_players
        self.n_cards = n_cards

        self.reset()

    def reset(self):
        self.index_visible_card = 0
        if self.n_cards == 150:  # regular game of skyjo
            self.pile = np.random.permutation(
                [-2] * 5 + [-1] * 10 + [0] * 15 + list(range(1, 13)) * 10
            )
        else:  # custom game of skyjo
            self.pile = np.random.permutation(
                list(range(-2, 13)) * (self.n_cards // 15) + [0] * (self.n_cards % 15)
            )
        self.players_deck_i_values = []
        self.players_deck_i_mask = []
        for _ in range(self.n_players):
            self.players_deck_i_values.append(
                np.reshape(
                    self.pile[self.index_visible_card : self.index_visible_card + 12],
                    self.grid_size,
                )
            )
            self.players_deck_i_mask.append(
                np.reshape([1] * 2 + [0] * 10, self.grid_size)
            )  # 2 visible cards at the beginning of a game
            self.index_visible_card += 12
        self.player_turn = [0, 0]
        self.done = False
        self.latest_action = None
        self.last_player = -1

        return self.get_observation()

    def get_observation(self, player_id=None):
        """Returns the current game state as an observation."""
        if player_id is None:
            player_id = self.player_turn[0]
        
        obs = {}
        obs["deck_values"] = self.players_deck_i_values[player_id]
        obs["deck_mask"] = self.players_deck_i_mask[player_id]
        obs["visible_card"] = self.pile[self.index_visible_card]
        obs["player_turn"] = self.player_turn

        return obs
    
    def get_state(self):
        state = {}
        state["player_turn"] = self.player_turn
        state["player_deck_i_values"] = self.players_deck_i_values
        state["player_deck_i_mask"] = self.players_deck_i_mask
        state["visible_card_value"] = self.pile[self.index_visible_card]
        state["done"] = self.done
        state["latest_action"] = self.latest_action
        
        return state

    def step(self, action):
        """Takes an action and updates the game state."""
        reward = 0
        player_id = self.player_turn[0]
        # TODO: reward ?
        if self.player_turn[1] == 0:  # player has to draw a card
            if action[0] == 0:  # player takes from the draw pile
                self.index_visible_card += 1  # the card drawn becomes the next visible card and will be used at next step
            self.player_turn[1] = 1

        # TODO: coder un shuffle du paquet plutôt que de mettre fin au jeu
        elif self.index_visible_card >= len(self.pile):
            self.done = True

        else:
            i, j = action[2:]
            # the player has drawn a card and will either discard it and turn over a card face down, or will replace any card from its deck by the card drawn
            if (
                action[1] == 1
            ):  # the player use the card drawn to replace one of its 12 cards 
                self.players_deck_i_mask[player_id][i,j] = 1
                replaced_card = self.players_deck_i_values[player_id][i,j]
                self.players_deck_i_values[player_id][i,j] = self.pile[
                    self.index_visible_card
                ]
                self.pile[self.index_visible_card] = replaced_card
            else:  # the player discards the card, and turn over a card face down
                if self.players_deck_i_mask[player_id][i,j] == 0:
                    self.players_deck_i_mask[player_id][i,j] = 1
                else:
                    raise ValueError(f"The card at the position {i,j} is already visible")

            if (
                np.sum(self.players_deck_i_mask[self.player_turn[0]]) == 12
                and self.last_player == -1
            ):
                self.last_player = (self.player_turn[0] - 1) % self.n_players

            if player_id == self.last_player:
                self.done = True

            self.player_turn = [(player_id + 1) % self.n_players, 0]

        return self.get_observation(), reward, self.done
