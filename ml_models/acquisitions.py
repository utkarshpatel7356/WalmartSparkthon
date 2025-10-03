import torch
import numpy as np
import os
from .actor_critic import Actor, Critic 

STATE_SIZE = 5
ACTION_SIZE = 1
MAX_ACTION = 1000
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor = Actor(STATE_SIZE, ACTION_SIZE, MAX_ACTION).to(device)
critic1 = Critic(STATE_SIZE, ACTION_SIZE).to(device)
critic2 = Critic(STATE_SIZE, ACTION_SIZE).to(device)

actor.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'checkpoint_actor_best.pth'), map_location=device))
critic1.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'checkpoint_critic1_best.pth'), map_location=device))
critic2.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'checkpoint_critic2_best.pth'), map_location=device))

actor.eval()
critic1.eval()
critic2.eval()

def predict_acquisition_quantity(input_data: dict) -> float:
    state_np = np.array([
        input_data["stock_level"],
        input_data["demand"],
        input_data["shelf_life_days"],
        input_data['price'],
        input_data['spoilage_cost']
    ], dtype=np.float32)

    state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(device)

    with torch.no_grad():
        action = actor(state_tensor)
        return float(action.cpu().numpy().flatten()[0])
