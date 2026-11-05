import copy
import json
import math
import random
import unittest

from poseidon.world import (ACTIONS, OBSERVATION_NAMES, TidePool, benchmark,
                            event_uniform, generate_transitions, rollout, teacher_action)


class WorldTests(unittest.TestCase):
    def test_exact_snapshot_continuation_through_json(self):
        env = TidePool(seed=173, max_steps=160)
        for _ in range(29):
            env.step(teacher_action(env.observe()))
        saved = json.loads(json.dumps(env.snapshot()))
        clone = TidePool.from_snapshot(saved)
        for _ in range(60):
            action = teacher_action(env.observe())
            self.assertEqual(env.step(action), clone.step(action))
            self.assertEqual(env.snapshot(), clone.snapshot())

    def test_observation_and_global_randomness_isolation(self):
        random.seed(71)
        before = random.getstate()
        one, two = TidePool(871), TidePool(871)
        for _ in range(30):
            obs = one.observe()
            self.assertEqual(len(obs), 16)
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in obs))
            self.assertEqual(obs, two.observe())
            one.observe()
            one.state()
            self.assertEqual(one.step(teacher_action(obs)), two.step(teacher_action(obs)))
        self.assertEqual(before, random.getstate())
        self.assertNotIn("teacher", " ".join(OBSERVATION_NAMES))

    def test_snapshot_import_is_transactional(self):
        env = TidePool(3)
        original = env.snapshot()
        for edit in (lambda s: s.update(health=float("nan")),
                     lambda s: s["patches"].pop(),
                     lambda s: s.update(done=True),
                     lambda s: s.update(visited=[999]),
                     lambda s: s.update(last_action=True)):
            bad = copy.deepcopy(original)
            edit(bad)
            with self.assertRaises(ValueError):
                env.load_snapshot(bad)
            self.assertEqual(original, env.snapshot())

    def test_survival_requires_useful_actions_and_horizon_is_not_death(self):
        seeds = list(range(91000001, 91000013))
        skilled = [rollout(teacher_action, seed, include_trajectory=False) for seed in seeds]
        idle = [rollout(lambda obs: 0, seed, include_trajectory=False) for seed in seeds]
        self.assertGreaterEqual(sum(r["survived"] for r in skilled), 10)
        self.assertEqual(sum(r["survived"] for r in idle), 0)
        self.assertTrue(all(r["truncated"] and r["terminal_reason"] == "horizon" for r in skilled if r["survived"]))
        self.assertTrue(all(r["death"] and r["death_reason"] for r in idle))
        single = TidePool(7, max_steps=1)
        _, _, ended, info = single.step(teacher_action(single.observe()))
        self.assertTrue(ended)
        self.assertTrue(info["truncated"])
        self.assertFalse(info["death"])
        with self.assertRaises(RuntimeError):
            single.step(0)

    def test_generator_is_reproducible_and_episode_disjoint(self):
        train = generate_transitions(420, seed=41)
        valid = generate_transitions(420, seed=42)
        self.assertEqual(train, generate_transitions(420, seed=41))
        self.assertFalse({r["episode_seed"] for r in train} & {r["episode_seed"] for r in valid})
        self.assertTrue(any(r["teacher_action"] != r["action"] for r in train))
        for row in train:
            self.assertEqual(row["teacher_action"], teacher_action(row["obs"]))
            self.assertTrue(all(0 <= x <= 1 for x in row["next_obs"]))
        first = train[0]
        env = TidePool(first["episode_seed"], first["scarcity"], max_steps=192)
        self.assertEqual(first["obs"], env.observe())
        self.assertEqual(first["next_obs"], env.step(first["action"])[0])

    def test_learned_policy_is_never_replaced_and_events_are_keyed(self):
        calls = []
        def invalid(obs):
            calls.append(obs)
            raise RuntimeError("Model unavailable")
        with self.assertRaisesRegex(RuntimeError, "Model unavailable"):
            rollout(invalid, 31)
        self.assertEqual(len(calls), 1)
        draw = event_uniform(3, 4, "weather", "rain")
        event_uniform(3, 99, "weather", "rain")
        self.assertEqual(draw, event_uniform(3, 4, "weather", "rain"))
        for bad in (-1, 6, True, 1.2):
            with self.assertRaises(ValueError):
                TidePool(1).step(bad)

    def test_paired_benchmark_retains_raw_rows(self):
        result = benchmark(teacher_action, seeds=[18, 19], max_steps=64)
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(len(result["paired_differences"]), 2)
        self.assertEqual(result["summary"]["learned"], result["summary"]["heuristic"])
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
