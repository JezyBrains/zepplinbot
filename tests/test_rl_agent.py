#!/usr/bin/env python3
"""
Tests for RL Agent Module
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rl_agent import CrashGameEnvironment, QLearningAgent, RLBettingAdvisor


class TestCrashGameEnvironment:
    """Test environment for RL training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample crash data."""
        np.random.seed(42)
        return np.exp(np.random.normal(0.5, 0.8, 200))
    
    @pytest.fixture
    def env(self, sample_data):
        """Create environment."""
        return CrashGameEnvironment(sample_data, initial_bankroll=100.0)
    
    def test_environment_initialization(self, env):
        """Test environment initializes correctly."""
        assert env.initial_bankroll == 100.0
        assert env.bankroll == 100.0
        assert not env.done
        assert env.n_actions == 7
    
    def test_environment_reset(self, env):
        """Test environment resets properly."""
        # Take a step
        env.step(1)
        
        # Reset
        state = env.reset()
        
        assert env.bankroll == env.initial_bankroll
        assert not env.done
        assert len(state) == 5  # State vector size
    
    def test_environment_step(self, env):
        """Test environment step returns correct format."""
        state = env.reset()
        
        next_state, reward, done, info = env.step(0)  # Skip action
        
        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_skip_action(self, env):
        """Test skip action doesn't change bankroll."""
        state = env.reset()
        initial = env.bankroll
        
        next_state, reward, done, info = env.step(0)  # Skip
        
        assert env.bankroll == initial
        assert info['result'] == 'skip'
    
    def test_bet_action(self, env):
        """Test bet action changes bankroll."""
        env.reset()
        
        next_state, reward, done, info = env.step(3)  # Medium bet
        
        assert info['result'] in ['win', 'loss']
        if info['result'] == 'win':
            assert info['profit'] > 0
        else:
            assert info['profit'] < 0


class TestQLearningAgent:
    """Test Q-learning agent."""
    
    @pytest.fixture
    def agent(self):
        """Create agent."""
        return QLearningAgent(
            n_actions=7,
            learning_rate=0.1,
            epsilon=1.0
        )
    
    @pytest.fixture
    def trained_agent(self, agent, sample_data):
        """Create and train agent."""
        env = CrashGameEnvironment(sample_data, initial_bankroll=100.0)
        agent.train(env, n_episodes=50, verbose=False)
        return agent
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.exp(np.random.normal(0.5, 0.8, 200))
    
    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.n_actions == 7
        assert agent.lr == 0.1
        assert agent.epsilon == 1.0
        assert len(agent.q_table) == 0  # Empty before training
    
    def test_agent_chooses_action(self, agent):
        """Test agent can choose actions."""
        state = np.array([1.0, 0.5, 0.8, 0, 0.3])
        
        action = agent.choose_action(state, training=True)
        
        assert 0 <= action < agent.n_actions
    
    def test_agent_updates_q_table(self, agent):
        """Test Q-table updates."""
        state = np.array([1.0, 0.5, 0.8, 0, 0.3])
        next_state = np.array([1.1, 0.6, 0.7, 0.1, 0.25])
        
        agent.update(state, action=1, reward=0.1, next_state=next_state, done=False)
        
        # Q-table should have entry now
        assert len(agent.q_table) > 0
    
    def test_agent_training(self, sample_data):
        """Test agent can be trained."""
        env = CrashGameEnvironment(sample_data, initial_bankroll=100.0)
        agent = QLearningAgent(n_actions=7, epsilon=1.0)
        
        stats = agent.train(env, n_episodes=10, verbose=False)
        
        assert 'n_states' in stats
        assert stats['n_states'] > 0
        assert len(agent.episode_rewards) == 10
    
    def test_epsilon_decay(self, sample_data):
        """Test epsilon decays during training."""
        env = CrashGameEnvironment(sample_data)
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.9)
        
        initial_epsilon = agent.epsilon
        agent.train(env, n_episodes=10, verbose=False)
        
        assert agent.epsilon < initial_epsilon
    
    def test_get_policy(self, trained_agent):
        """Test policy extraction."""
        state = np.array([1.0, 0.5, 0.8, 0, 0.3])
        
        policy = trained_agent.get_policy(state)
        
        assert 'best_action' in policy
        assert 'action_name' in policy
        assert 'q_values' in policy
        assert 0 <= policy['best_action'] < 7


class TestRLBettingAdvisor:
    """Test RL-based betting advisor."""
    
    @pytest.fixture
    def advisor(self):
        agent = QLearningAgent(n_actions=7)
        return RLBettingAdvisor(agent)
    
    def test_advisor_recommendation(self, advisor):
        """Test advisor gives recommendation."""
        recent = np.random.exponential(2, 30)
        
        rec = advisor.get_recommendation(recent, bankroll=100.0)
        
        assert 'action' in rec
        assert rec['action'] in ['skip', 'bet']
    
    def test_advisor_with_insufficient_data(self, advisor):
        """Test advisor handles insufficient data."""
        short_data = np.array([2.0, 3.0, 1.5])
        
        rec = advisor.get_recommendation(short_data, 100.0)
        
        assert rec['action'] == 'skip'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
