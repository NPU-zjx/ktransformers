"""Deterministic global PPO minibatches; old probabilities live for a whole rollout batch."""
import random


def schedule(rows, minibatch_responses, epochs, *, shuffle=False, seed=42):
    if not rows or len(rows) % minibatch_responses:
        raise ValueError("Rollout batch must contain whole PPO minibatches")
    result = []
    for epoch in range(epochs):
        order = list(range(len(rows)))
        if shuffle:
            random.Random(seed + epoch).shuffle(order)
        for mini, start in enumerate(range(0, len(order), minibatch_responses)):
            indices = order[start:start + minibatch_responses]
            tokens = sum(len(rows[i]["response"]) for i in indices)
            if tokens <= 0:
                raise ValueError("Empty response minibatch")
            result.extend((i, epoch, mini, tokens) for i in indices)
    return result
