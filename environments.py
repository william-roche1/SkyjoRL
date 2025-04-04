import numpy as np
import torch

class SkyjoEnv:
    def __init__(self, n_players=4, n_cards=150, device=torch.device("cpu")):
        self.grid_size = [3, 4]
        self.n_players = n_players
        self.n_cards = n_cards
        self.device = device

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

    def get_valid_actions(self, player_turn, deck_mask, device=torch.device("cpu")):
        if player_turn[1] == 0: # Player has to draw a card
            return [torch.tensor([0,-1,-1,-1], device=device), torch.tensor([1,-1,-1,-1], device=device)]
        else:
            valid_actions = []
            for i in range(deck_mask.shape[0]):
                for j in range(deck_mask.shape[1]):
                    valid_actions.append(torch.tensor([-1,1,i,j], device=device))
                    if deck_mask[i,j] == 0:
                        valid_actions.append(torch.tensor([-1,0,i,j], device=device))
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
    
        valid_actions = self.get_valid_actions(self.player_turn, deck_masks, device=self.device)
        if not any(torch.equal(action, x) for x in valid_actions):
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








# Column Environment

# Define a custom cards order
def define_custom_pile(first_cards):
    number_of_cards = 10 * np.ones(15, dtype=int)
    number_of_cards[0] = 5
    number_of_cards[2] = 15
    pile = np.zeros(150)
    for i, card in enumerate(first_cards):
        pile[i] = card
        number_of_cards[card] -= 1
    card_values = np.arange(-2,13,1)
    remaining_cards = np.repeat(card_values, number_of_cards)
    pile[len(first_cards):] = np.random.permutation(remaining_cards)
    return pile.astype(int)


# Column Skyjo Env
class ColumnSkyjo:
    def __init__(self, grid_len=3, max_rounds=10, initial_visible_cards=0):
        self.grid_len = grid_len
        self.max_rounds = max_rounds
        self.initial_visible_cards = initial_visible_cards
        assert initial_visible_cards <= grid_len

        self.reset()

    def reset(self, custom_pile=None, initial_visible_cards=None):
        self.round_id = 0
        self.drawn_card_id = self.grid_len
        if initial_visible_cards is not None:
            self.initial_visible_cards = initial_visible_cards
        if custom_pile is None:
            self.pile = np.random.permutation([-2] * 5 + [-1] * 10 + [0] * 15 + list(range(1, 13)) * 10)
        else:
            self.pile = custom_pile.copy()
            print("Custom pile: ", self.pile[:10])
        self.deck_values = self.pile[:self.grid_len]
        self.deck_mask = np.zeros(self.grid_len, dtype=bool)
        self.deck_mask[:self.initial_visible_cards] = np.ones(self.initial_visible_cards, dtype=bool)
        self.done = False
        self.argsorted_indices = np.arange(self.grid_len)

        return self.get_observation()
    
    def get_valid_actions(self):
        valid_actions = [0,1,2,3]
        return valid_actions

    def get_observation(self):
        """
        Returns the current game state as an observation for a specific player id.
        If no player_id is specified, return the observation for the player who needs to play.
        """        
        deck = self.deck_values.copy()
        deck[~self.deck_mask] = -3
        argsorted_indices = np.argsort(deck)
        deck_sorted = deck[argsorted_indices]
        cards = np.append(deck_sorted, self.pile[self.drawn_card_id])

        return cards
    
    def get_state(self):  
        state = {}
        state["deck_values"] = self.deck_values.copy()
        state["deck_mask"] = self.deck_mask.copy()
        state["drawn_card"] = self.pile[self.drawn_card_id]
        state["round_id"] = self.round_id
        state["done"] = self.done

        return state
    
    def get_sum_cards(self, deck_mask=np.ones((3), dtype=bool)):
        return np.sum(self.deck_values[deck_mask.astype(bool)]) + 5.0667 * np.sum((~deck_mask.astype(bool)).astype(int))

    def step(self, action):
        """Takes an action and updates the game state."""

        deck = self.deck_values.copy()
        deck[~self.deck_mask] = -3
        self.argsorted_indices = np.argsort(deck)
        
        if action > 0:
            action = 1 + self.argsorted_indices[action - 1]

        deck_values = self.deck_values
        deck_mask = self.deck_mask

        self.round_id += 1
        sum_cards_before = self.get_sum_cards(deck_mask)
    
        valid_actions = self.get_valid_actions()
        if not (action in valid_actions):
            raise ValueError(f"Unauthorized action: {action}")
        
        if action == 0:
            if not deck_mask[0]:
                deck_mask[0] = 1
            elif not deck_mask[1]:
                deck_mask[1] = 1
            elif not deck_mask[2]:
                deck_mask[2] = 1
            self.drawn_card_id += 1
        else:
            drawn_card = self.pile[self.drawn_card_id]
            discarded_card = deck_values[action - 1]
            self.pile[self.drawn_card_id] = discarded_card
            deck_values[action - 1] = drawn_card
            deck_mask[action - 1] = 1
            self.drawn_card_id += 1
        
        # Drop a column if there is 3 times the same card
        if np.all(deck_mask) and np.all(deck_values == deck_values[0]):
            deck_values *= 0

        if self.drawn_card_id >= len(self.pile):
            self.drawn_card_id = self.grid_len
            self.pile[self.drawn_card_id:] = np.random.permutation(self.pile[self.drawn_card_id:])

        sum_cards_after = self.get_sum_cards(deck_mask)

        reward = sum_cards_before - sum_cards_after - 1/self.max_rounds
        reward = 0

        if (sum_cards_after <= 0 and all(deck_mask)) or self.round_id == self.max_rounds:
            self.done = True
            reward = -sum_cards_after
        
        return self.get_observation(), reward, self.done
