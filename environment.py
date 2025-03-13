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
                    (4, 3),
                )
            )
            self.players_deck_i_mask.append(
                np.reshape([1] * 2 + [0] * 10, (3, 4))
            )  # 2 visible cards at the beginning of a game
            self.index_visible_card += 12
        self.player_turn = (0, 0)
        self.done = False
        self.last_player = -1

        return self.get_observation()

    def get_observation(self):
        """Returns the current game state as an observation."""
        return (
            self.players_deck_i_values[self.player_turn[0]],
            self.players_deck_i_mask[self.player_turn[0]],
            self.pile[self.index_visible_card],
            self.player_turn,
        )

    def step(self, action):
        """Takes an action and updates the game state."""
        reward = 0
        # TODO: reward ?
        if self.player_turn[1] == 0:  # player has to draw a card
            if action == 0:  # player takes from the draw pile
                self.index_visible_card += 1  # the card drawn becomes the next visible card and will be used at next step
            self.player_turn[1] += 1

        # TODO: coder un shuffle du paquet plutôt que de mettre fin au jeu
        if self.index_visible_card >= len(self.pile):
            self.done = True

        else:
            # the player will take the visible card and put it in one of its 12 positions or back in the discard pile
            if (
                action >= 0
            ):  # the player takes the visible card and put it in one of its 12 positions
                i, j = action // 3, action % 3
                self.players_deck_i_mask[self.player_turn[0]][j][i] == 1
                replaced_card = self.players_deck_i_values[self.player_turn[0]][j][i]
                self.players_deck_i_values[self.player_turn[0]][j][i] = self.pile[
                    self.index_visible_card
                ]
                self.pile[self.index_visible_card] = replaced_card
            else:  # the player puts the visible card back in the discard pile
                i, j = (-action - 1) // 3, (-action - 1) % 3
                if self.players_deck_i_mask[self.player_turn[0]][j][i] == 0:
                    self.players_deck_i_mask[self.player_turn[0]][j][i] = 1
                else:
                    raise ValueError("The card is already visible")

            if (
                np.sum(self.players_deck_i_mask[self.player_turn[0]]) == 12
                and self.last_player == -1
            ):
                self.last_player = (self.player_turn[0] - 1) % 4

            if self.player_turn[0] == self.last_player:
                self.done = True

            self.player_turn = (self.player_turn[0] + 1) % self.n_players, 0

        return self.get_observation(), reward, self.done
