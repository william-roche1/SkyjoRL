import numpy as np
import torch

class SkyjoEnv:
    def __init__(self, n_players=4, n_cards=150):
        self.grid_size = [3, 4]
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

    def get_valid_actions(self):
        if self.player_turn[1] == 0: # Player has to draw a card
            return [torch.tensor([0,-1,-1,-1]), torch.tensor([1,-1,-1,-1])]
        else:
            valid_actions = []
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    valid_actions.append(torch.tensor([-1,1,i,j]))
                    if self.players_deck_i_mask[self.player_turn[0]][i,j] == 0:
                        valid_actions.append(torch.tensor([-1,0,i,j]))
            return valid_actions

    
    def get_sum_cards(self, deck_values, deck_masks):
        deck_masks_bool = deck_masks.astype(bool)
        sum_visible = np.sum(deck_values[deck_masks_bool])
        sum_invisible = np.sum(1 - deck_masks.astype(int)) * 5.0667 # When the card is invisible, we assign the average value (np.mean(self.pile))
        return sum_visible + sum_invisible


    def get_observation(self, player_id=None):
        """
        Returns the current game state as an observation for a specific player id.
        If no player_id is specified, return the observation for the player who needs to play.
        """
        if player_id is None:
            player_id = self.player_turn[0]
        
        obs = {}
        obs["deck_values"] = self.players_deck_i_values[player_id].copy()
        obs["deck_mask"] = self.players_deck_i_mask[player_id].copy()
        obs["visible_card"] = self.pile[self.index_visible_card]
        obs["player_turn"] = self.player_turn.copy()

        return obs
    
    def get_state(self):
        state = {}
        state["player_turn"] = self.player_turn.copy()
        state["player_deck_i_values"] = self.players_deck_i_values.copy()
        state["player_deck_i_mask"] = self.players_deck_i_mask.copy()
        state["visible_card_value"] = self.pile[self.index_visible_card]
        state["done"] = self.done
        state["latest_action"] = self.latest_action
        
        return state

    def step(self, action):
        """Takes an action and updates the game state."""
        player_id = self.player_turn[0]
        deck_values = self.players_deck_i_values[player_id]
        deck_masks = self.players_deck_i_mask[player_id]
    
        valid_actions = self.get_valid_actions()
        if not any(np.array_equal(action, x) for x in valid_actions):
            raise ValueError(f"Unauthorized action: {action}")

        if self.player_turn[1] == 0:  # player has to draw a card
            if action[0] == 0:  # player takes from the draw pile
                self.index_visible_card += 1  # the card drawn becomes the next visible card and will be used at next step
            self.player_turn[1] = 1
            reward = 0 # Reward is set to 0 

        else:
            sum_before_playing = self.get_sum_cards(deck_values, deck_masks)
            i, j = action[2:]
            # the player has drawn a card and will either discard it and turn over a card face down, or will replace any card from its deck by the card drawn
            if (
                action[1] == 1
            ):  # the player use the card drawn to replace one of its 12 cards 
                deck_masks[i,j] = 1
                replaced_card = deck_values[i,j]
                deck_values[i,j] = self.pile[
                    self.index_visible_card
                ]
                self.pile[self.index_visible_card] = replaced_card
            else:  # the player discards the card, and turn over a card face down
                if deck_masks[i,j] == 0:
                    deck_masks[i,j] = 1
                else:
                    raise ValueError(f"The card at the position {i,j} is already visible")
            
            # Drop a column if there is 3 times the same card
            if np.all(deck_masks[:,j]) and np.all(deck_values[:,j] == deck_values[0,j]):
                deck_values[:,j] *= 0

            sum_after_playing = self.get_sum_cards(deck_values, deck_masks)
            reward = -(sum_after_playing - sum_before_playing)

            if (
                np.sum(deck_masks) == 12
                and self.last_player == -1
            ):
                self.last_player = (player_id - 1) % self.n_players

            if player_id == self.last_player:
                self.done = True
            
            self.player_turn = [(player_id + 1) % self.n_players, 0]

        # Shuffle the cards if the draw pile is empty
        if self.index_visible_card >= len(self.pile):
            self.index_visible_card = self.n_players * 12
            self.pile[self.index_visible_card:] = np.random.permutation(self.pile[self.index_visible_card:])
        
        return self.get_observation(), reward, self.done
