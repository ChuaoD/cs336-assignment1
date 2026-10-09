import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from cs336_basics import train_lm
from cs336_basics.data_loader import load_data


def test_validation_generator_is_repeatable_and_independent():
    dataset = np.arange(100)
    training_state = np.random.get_state()
    first = load_data(dataset, 8, 4, "cpu", rng=np.random.default_rng(123))
    second = load_data(dataset, 8, 4, "cpu", rng=np.random.default_rng(123))
    after_validation = np.random.get_state()

    assert training_state[0] == after_validation[0]
    np.testing.assert_array_equal(training_state[1], after_validation[1])
    assert training_state[2:] == after_validation[2:]
    for left, right in zip(first, second):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(first[1], first[0] + 1)


def test_training_reuses_validation_without_changing_training_samples(tmp_path, monkeypatch):
    dataset_path = tmp_path / "tokens.bin"
    (np.arange(128) % 16).astype(np.uint16).tofile(dataset_path)
    config = json.loads((Path(__file__).resolve().parents[1] / "configs.json").read_text())
    config["data"].update(train_path=str(dataset_path), validation_path=str(dataset_path))
    config["model"].update(
        vocab_size=16, context_length=4, d_model=8, num_layers=1, num_heads=2, d_ff=16
    )
    config["training"].update(batch_size=2, max_iters=3, device="cpu")
    config["evaluation"].update(interval=1, num_eval_batches=2)
    config["scheduler"].update(warmup_iters=1, cosine_cycle_iters=3)
    config["checkpoint"].update(output_dir=str(tmp_path / "checkpoints"), interval=3, resume_from=None)
    config["logging"].update(use_wandb=False, interval=1)
    config_path = tmp_path / "config.json"
    recorded_models = []
    logged_configs = []

    class RecordingTransformer(train_lm.TransformerLM):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.initial_weights = {name: value.clone() for name, value in self.state_dict().items()}
            self.train_inputs = []
            self.validation_inputs = []
            recorded_models.append(self)

        def forward(self, inputs):
            records = self.train_inputs if self.training else self.validation_inputs
            records.append(inputs.detach().clone())
            return super().forward(inputs)

    monkeypatch.setattr(train_lm, "TransformerLM", RecordingTransformer)
    monkeypatch.setattr(
        train_lm, "wandb",
        SimpleNamespace(init=lambda **kwargs: logged_configs.append(kwargs["config"]), log=lambda _: None),
    )

    for learning_rate, num_eval_batches in [(0.001, 2), (0.003, 2), (0.001, 3)]:
        config["evaluation"]["num_eval_batches"] = num_eval_batches
        config_path.write_text(json.dumps(config))
        train_lm.train(config_path, learning_rate=learning_rate)

    first, second, third = recorded_models
    for other in (second, third):
        for name, initial in first.initial_weights.items():
            torch.testing.assert_close(initial, other.initial_weights[name], rtol=0, atol=0)
        for left, right in zip(first.train_inputs, other.train_inputs):
            torch.testing.assert_close(left, right)
    for model, batch_count in zip(recorded_models, (2, 2, 3)):
        assert len(model.train_inputs) == 3
        assert len(model.validation_inputs) == 3 * batch_count
        for index, inputs in enumerate(model.validation_inputs):
            torch.testing.assert_close(inputs, model.validation_inputs[index % batch_count])
    for left, right in zip(first.validation_inputs, second.validation_inputs):
        torch.testing.assert_close(left, right)
    for logged in logged_configs:
        assert logged["training_seed"] == 42
        assert logged["evaluation_seed"] == 123
