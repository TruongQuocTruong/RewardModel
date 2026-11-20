"""CPU lifecycle tests: policy starts after reward readiness and owns cleanup."""
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch

import run_job


class JobTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent / '_verification' / self._testMethodName
        self.root.mkdir(parents=True)
        reward_env = self.root / 'reward_env'
        (reward_env / 'bin').mkdir(parents=True)
        (reward_env / 'bin/python').write_text('# fixture')
        self.settings = {'reward_env': str(reward_env), 'write_probe_mib': 0, 'reward_ready_timeout': 10}
        (self.root / 'settings.json').write_text(json.dumps(self.settings))
        self.run = self.root / 'job_1'
    def tearDown(self):
        shutil.rmtree(self.root)
    def prepare(self, *args, **kwargs):
        self.run.mkdir()
        (self.run / 'settings.json').write_text(json.dumps({**self.settings, 'port': 18888, 'run_id': str(self.run)}))

    def run_case(self, policy_exit):
        reward, policy = Mock(), Mock()
        reward.poll.return_value = None
        policy.poll.side_effect = [None, policy_exit, policy_exit]
        policy.returncode = policy_exit
        with patch.object(run_job.sys, 'argv', ['run_job.py', '--bundle', str(self.root), '--run-dir', str(self.run)]), \
             patch.object(run_job.subprocess, 'run', side_effect=self.prepare), \
             patch.object(run_job.subprocess, 'Popen', side_effect=[reward, policy]) as popen, \
             patch.object(run_job, 'wait_ready') as ready, \
             patch.object(run_job.time, 'sleep'), \
             patch.object(run_job.signal, 'signal'):
            if policy_exit:
                with self.assertRaisesRegex(RuntimeError, 'Policy exited'):
                    run_job.main()
            else:
                run_job.main()
            self.assertEqual(ready.call_args.args[0]['run_id'], str(self.run))
            self.assertEqual(len(popen.call_args_list), 2)
            self.assertEqual(Path(popen.call_args_list[1].args[0][2]).name, 'train.py')
        reward.terminate.assert_called_once()
        policy.terminate.assert_not_called()

    def test_success_stops_reward(self):
        self.run_case(0)

    def test_policy_failure_stops_reward(self):
        self.run_case(1)


if __name__ == '__main__':
    unittest.main()
