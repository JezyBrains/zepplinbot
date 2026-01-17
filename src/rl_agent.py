#!/usr/bin/env python3
"""
Reinforcement Learning Agent for Crash Game Betting

Implements Q-learning to learn optimal betting policies.
State space includes volatility, bankroll ratio, and streak info.
Actions: skip, bet_small, bet_medium, bet_large.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrashGameEnvironment:
    """
    Environment for RL agent training on crash game data.
    
    Simulates betting rounds using historical data.
    """
    
    def __init__(self, 
                 crash_data: np.ndarray,
                 initial_bankroll: float = 100.0):
        """
        Args:
            crash_data: Array of historical crash values
            initial_bankroll: Starting bankroll
        """
        self.crash_data = crash_data
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.current_idx = 50  # Start after enough history
        self.done = False
        
        # Action definitions: (bet_fraction, target_multiplier)
        self.actions = {
            0: (0.0, 0.0),      # Skip
            1: (0.02, 1.5),     # Small bet, low target
            2: (0.03, 2.0),     # Small bet, medium target
            3: (0.05, 1.5),     # Medium bet, low target
            4: (0.05, 2.0),     # Medium bet, medium target
            5: (0.08, 2.0),     # Large bet, medium target
            6: (0.08, 3.0),     # Large bet, high target
        }
        self.n_actions = len(self.actions)
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.bankroll = self.initial_bankroll
        self.current_idx = 50
        self.done = False
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get current state vector.
        
        State features:
        - Bankroll ratio (current / initial)
        - Recent volatility (std of last 20 crashes)
        - Recent mean crash (last 20)
        - Streak indicator (consecutive lows/highs)
        - Crash below 1.5x ratio (last 10)
        """
        recent = self.crash_data[self.current_idx-20:self.current_idx]
        last_10 = self.crash_data[self.current_idx-10:self.current_idx]
        
        bankroll_ratio = self.bankroll / self.initial_bankroll
        volatility = np.std(recent) / np.mean(recent) if np.mean(recent) > 0 else 0
        mean_crash = np.mean(recent)
        low_ratio = np.mean(last_10 < 1.5)  # Fraction under 1.5x
        
        # Detect streak
        streak = 0
        for i in range(len(last_10) - 1, -1, -1):
            if last_10[i] < 1.5:
                streak -= 1
            elif last_10[i] > 3.0:
                streak += 1
            else:
                break
        streak = np.clip(streak / 5, -1, 1)  # Normalize
        
        return np.array([
            np.clip(bankroll_ratio, 0, 3),  # Cap at 3x initial
            np.clip(volatility, 0, 2),
            np.clip(mean_crash / 5, 0, 2),  # Normalize
            streak,
            low_ratio
        ])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action and observe result.
        
        Args:
            action: Action index
        
        Returns:
            (next_state, reward, done, info)
        """
        bet_fraction, target = self.actions[action]
        crash = self.crash_data[self.current_idx]
        
        reward = 0
        info = {'action': action, 'crash': crash, 'target': target}
        
        if bet_fraction > 0:
            bet_amount = self.bankroll * bet_fraction
            
            if crash >= target:
                # Win
                profit = bet_amount * (target - 1)
                self.bankroll += profit
                reward = profit / self.initial_bankroll  # Normalize reward
                info['result'] = 'win'
                info['profit'] = profit
            else:
                # Lose
                self.bankroll -= bet_amount
                reward = -bet_amount / self.initial_bankroll
                info['result'] = 'loss'
                info['profit'] = -bet_amount
        else:
            info['result'] = 'skip'
            info['profit'] = 0
            reward = 0  # Small penalty for skipping could be added
        
        self.current_idx += 1
        self.done = self.current_idx >= len(self.crash_data) - 1 or self.bankroll <= 0
        
        next_state = self._get_state()
        return next_state, reward, self.done, info


class QLearningAgent:
    """
    Q-learning agent for crash game betting.
    
    Uses discretized state space for tabular Q-learning.
    """
    
    def __init__(self,
                 n_actions: int = 7,
                 state_bins: List[int] = None,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01):
        """
        Args:
            n_actions: Number of possible actions
            state_bins: Bins for discretizing each state feature
            learning_rate: Q-learning alpha
            discount_factor: Q-learning gamma
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay per episode
            epsilon_min: Minimum epsilon
        """
        self.n_actions = n_actions
        self.state_bins = state_bins or [10, 10, 10, 5, 5]
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _discretize_state(self, state: np.ndarray) -> Tuple:
        """Convert continuous state to discrete bins."""
        discrete = []
        for i, (val, bins) in enumerate(zip(state, self.state_bins)):
            # Normalize to [0, 1] range (assuming state features are roughly in [0, 2])
            normalized = np.clip(val / 2, 0, 0.999)
            bin_idx = int(normalized * bins)
            discrete.append(bin_idx)
        return tuple(discrete)
    
    def choose_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether to use exploration
        
        Returns:
            Action index
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        discrete_state = self._discretize_state(state)
        return np.argmax(self.q_table[discrete_state])
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) <- Q(s,a) + α * (r + γ * max_a' Q(s',a') - Q(s,a))
        """
        discrete_state = self._discretize_state(state)
        discrete_next = self._discretize_state(next_state)
        
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next])
        
        self.q_table[discrete_state][action] += self.lr * (target - current_q)
    
    def train(self, env: CrashGameEnvironment, n_episodes: int = 1000, 
              verbose: bool = True) -> Dict:
        """
        Train agent on environment.
        
        Args:
            env: CrashGameEnvironment instance
            n_episodes: Number of training episodes
            verbose: Print progress
        
        Returns:
            Training statistics
        """
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            steps = 0
            
            while not env.done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(f"Episode {episode + 1}: Avg Reward = {avg_reward:.4f}, "
                          f"Epsilon = {self.epsilon:.3f}, States = {len(self.q_table)}")
        
        return {
            'final_epsilon': self.epsilon,
            'n_states': len(self.q_table),
            'avg_reward_last_100': np.mean(self.episode_rewards[-100:]),
            'max_reward': max(self.episode_rewards)
        }
    
    def get_policy(self, state: np.ndarray) -> Dict:
        """
        Get policy recommendation for a state.
        
        Returns:
            Dict with action and Q-values
        """
        discrete_state = self._discretize_state(state)
        q_values = self.q_table[discrete_state]
        best_action = np.argmax(q_values)
        
        action_names = ['skip', 'small_1.5x', 'small_2.0x', 'medium_1.5x', 
                       'medium_2.0x', 'large_2.0x', 'large_3.0x']
        
        return {
            'best_action': best_action,
            'action_name': action_names[best_action],
            'q_values': {name: q_values[i] for i, name in enumerate(action_names)},
            'confidence': np.max(q_values) - np.mean(q_values)
        }
    
    def get_confidence_score(self, state: np.ndarray) -> float:
        """
        Calculate confidence score for current state (V2).
        
        Uses the Q-value distribution to determine how confident
        the agent is in its recommended action.
        
        Returns:
            Confidence score in range [0, 1]
        """
        discrete_state = self._discretize_state(state)
        q_values = self.q_table[discrete_state]
        
        if np.all(q_values == 0):
            return 0.5  # Unknown state, neutral confidence
        
        # Softmax-based confidence
        exp_q = np.exp(q_values - np.max(q_values))  # Numerical stability
        softmax = exp_q / np.sum(exp_q)
        
        # Confidence = max probability (how certain is the best action)
        confidence = np.max(softmax)
        
        return float(np.clip(confidence, 0, 1))
    
    def get_kelly_bet(self, state: np.ndarray, bankroll: float, 
                      target: float = 2.0, kelly_fraction: float = 0.25) -> Dict:
        """
        Calculate Kelly Criterion bet size adjusted by confidence (V2).
        
        f* = (bp - q) / b
        where:
        - b = target - 1 (net odds)
        - p = estimated win probability (from confidence)
        - q = 1 - p
        
        Uses fractional Kelly (default 0.25x) for risk management.
        
        Args:
            state: Current state vector
            bankroll: Current bankroll
            target: Target multiplier
            kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
        
        Returns:
            Dict with bet sizing information
        """
        confidence = self.get_confidence_score(state)
        
        # Map confidence to win probability
        # Confidence > 0.6 suggests favorable conditions
        win_prob = 0.4 + (confidence - 0.5) * 0.4  # Maps [0.5, 1] -> [0.4, 0.6]
        win_prob = np.clip(win_prob, 0.3, 0.7)
        
        b = target - 1  # Net odds
        q = 1 - win_prob
        
        # Kelly formula
        kelly = (b * win_prob - q) / b if b > 0 else 0
        kelly = max(0, kelly)  # No negative bets
        
        # Apply fractional Kelly
        adjusted_kelly = kelly * kelly_fraction
        bet_amount = bankroll * adjusted_kelly
        
        return {
            'confidence': round(confidence, 3),
            'win_probability': round(win_prob, 3),
            'kelly_full': round(kelly, 4),
            'kelly_fraction': kelly_fraction,
            'kelly_adjusted': round(adjusted_kelly, 4),
            'bet_amount': round(bet_amount, 2),
            'bet_percentage': round(adjusted_kelly * 100, 2)
        }
    
    def save(self, filepath: str):
        """Save agent to file."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = defaultdict(lambda: np.zeros(self.n_actions), data['q_table'])
        self.epsilon = data['epsilon']
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_lengths = data.get('episode_lengths', [])
        logger.info(f"Agent loaded from {filepath}")


class RLBettingAdvisor:
    """
    High-level advisor using trained RL agent.
    """
    
    def __init__(self, agent: QLearningAgent):
        self.agent = agent
        self.action_details = {
            0: {'action': 'skip', 'bet_fraction': 0, 'target': 0},
            1: {'action': 'bet', 'bet_fraction': 0.02, 'target': 1.5},
            2: {'action': 'bet', 'bet_fraction': 0.03, 'target': 2.0},
            3: {'action': 'bet', 'bet_fraction': 0.05, 'target': 1.5},
            4: {'action': 'bet', 'bet_fraction': 0.05, 'target': 2.0},
            5: {'action': 'bet', 'bet_fraction': 0.08, 'target': 2.0},
            6: {'action': 'bet', 'bet_fraction': 0.08, 'target': 3.0},
        }
    
    def get_recommendation(self, 
                          recent_crashes: np.ndarray,
                          bankroll: float,
                          initial_bankroll: float = 100.0) -> Dict:
        """
        Get betting recommendation from RL agent.
        
        Args:
            recent_crashes: Last 20+ crash values
            bankroll: Current bankroll
            initial_bankroll: Starting bankroll
        
        Returns:
            Recommendation dict
        """
        if len(recent_crashes) < 20:
            return {'action': 'skip', 'reason': 'Insufficient data'}
        
        # Build state
        last_20 = recent_crashes[-20:]
        last_10 = recent_crashes[-10:]
        
        bankroll_ratio = bankroll / initial_bankroll
        volatility = np.std(last_20) / np.mean(last_20)
        mean_crash = np.mean(last_20)
        low_ratio = np.mean(last_10 < 1.5)
        
        streak = 0
        for i in range(len(last_10) - 1, -1, -1):
            if last_10[i] < 1.5:
                streak -= 1
            elif last_10[i] > 3.0:
                streak += 1
            else:
                break
        streak = np.clip(streak / 5, -1, 1)
        
        state = np.array([
            np.clip(bankroll_ratio, 0, 3),
            np.clip(volatility, 0, 2),
            np.clip(mean_crash / 5, 0, 2),
            streak,
            low_ratio
        ])
        
        policy = self.agent.get_policy(state)
        action_idx = policy['best_action']
        details = self.action_details[action_idx]
        
        bet_amount = bankroll * details['bet_fraction']
        
        return {
            'action': details['action'],
            'bet_amount': bet_amount,
            'bet_fraction': details['bet_fraction'] * 100,
            'target_multiplier': details['target'],
            'confidence': policy['confidence'],
            'q_values': policy['q_values'],
            'state_info': {
                'bankroll_ratio': bankroll_ratio,
                'volatility': volatility,
                'mean_crash': mean_crash,
                'low_ratio': low_ratio
            }
        }


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("REINFORCEMENT LEARNING AGENT")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/zeppelin_data.csv')
    data = df['value'].values
    
    # Create environment and agent
    env = CrashGameEnvironment(data, initial_bankroll=100.0)
    agent = QLearningAgent(n_actions=7)
    
    # Train
    print("\n🎓 Training agent (1000 episodes)...")
    stats = agent.train(env, n_episodes=1000, verbose=True)
    
    print(f"\n📊 Training Complete:")
    print(f"   Final Epsilon: {stats['final_epsilon']:.4f}")
    print(f"   States Explored: {stats['n_states']}")
    print(f"   Avg Reward (last 100): {stats['avg_reward_last_100']:.4f}")
    
    # Test policy
    print("\n" + "=" * 60)
    print("POLICY EVALUATION")
    print("=" * 60)
    
    advisor = RLBettingAdvisor(agent)
    
    # Get recommendation for current state
    recent = data[-30:]
    rec = advisor.get_recommendation(recent, bankroll=100.0)
    
    print(f"\n🎯 Recommendation:")
    print(f"   Action: {rec['action'].upper()}")
    if rec['action'] == 'bet':
        print(f"   Bet Amount: ${rec['bet_amount']:.2f} ({rec['bet_fraction']:.1f}%)")
        print(f"   Target: {rec['target_multiplier']}x")
    print(f"   Confidence: {rec['confidence']:.4f}")
    
    print(f"\n📈 Q-Values:")
    for action, qval in rec['q_values'].items():
        bar = '█' * int(max(0, qval * 10 + 5))
        print(f"   {action:12s}: {bar} {qval:.4f}")
    
    # Save agent
    agent.save('models/saved/rl_agent.pkl')
