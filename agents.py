import numpy as np

# Baseline agent to evaluate our other agents' performance

class BaselineAgent:
    def __init__(self, player_id, visible_card_threshold=3, discard_card_threshold=6, flip_card_proba=0.2):
        self.player_id = player_id
        self.visible_card_threshold = visible_card_threshold
        self.discard_card_threshold = discard_card_threshold
        self.flip_card_proba = flip_card_proba

        self.latest_action = None
    
    def get_action(self, observation: dict):
        """From the observation, returns the action chosen by the agent."""
        deck_values = observation["deck_values"]
        deck_mask = observation["deck_mask"]
        visible_card = observation["visible_card"]
        player_turn = observation["player_turn"]

        if player_turn[0] != self.player_id: # Raise an error if we ask our agent to player and its not its turn to player
            raise ValueError(f"It's not my turn to play! I am player {self.player_id}, whereas player {player_turn[0]} needs to play.")
        
        if player_turn[1] == 0: # Our agent needs to choose wether to draw the visible card, or from the draw pile
            if visible_card <= self.visible_card_threshold: # Draw the visible card if its value is lower than a threshold
                action = [1,-1,-1,-1]
            else:
                action = [0,-1,-1,-1]

        else: # Our agent needs to decide wether to discard the card or not, and then choose a position
            indices_facedown_cards = np.argwhere(deck_mask == False) # Indices of the face-down cards            
            i, j = indices_facedown_cards[np.random.randint(0, indices_facedown_cards.shape[0])] # Get some random face-down card
            if np.random.rand() < self.flip_card_proba: # We force our agent to choose to flip a random face-down card with some probability

                if visible_card >= self.discard_card_threshold: # Discard the drawn card if its value is higher or equal than a threshold
                    action = [-1,0,i,j]
                else:   # Keep the card
                    action = [-1,1,i,j]
            else:
                max_value_deck = np.max(deck_values[deck_mask])
                if max_value_deck > visible_card: # Keep the card, and replace it by the highest card in the deck
                    id_max = list(zip(*np.where(deck_values == max_value_deck)))[0]
                    action = [-1,1,id_max[0],id_max[1]]
                else:   # Discard the card, and flip a random face-down card
                    action = [-1,0,i,j]
        
        return action

    def get_latest_action(self):
        return self.latest_action