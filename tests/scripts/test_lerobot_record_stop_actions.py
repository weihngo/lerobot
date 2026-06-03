import json
from collections import deque
from types import SimpleNamespace

from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts import lerobot_record


class RecordingRobot:
    def __init__(self, events: dict[str, bool]):
        self.events = events
        self.name = "lekiwi_client"
        self.robot_type = "lekiwi_client"
        self.action_features = {
            "arm_shoulder_pan.pos": float,
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }
        self.sent_actions: list[dict[str, float]] = []

    def get_observation(self):
        return {"observation.state": 1.0}

    def send_action(self, action):
        self.sent_actions.append(dict(action))
        if len(self.sent_actions) == 3:
            self.events["exit_early"] = True
        return action


class ChunkPolicy:
    def __init__(self):
        self.config = SimpleNamespace(device="cpu", use_amp=False)
        self.reset()

    def reset(self):
        self._action_queue = deque([object(), object()])


class DummyDataset:
    fps = 30
    features = {}

    def __init__(self):
        self.frames = []

    def add_frame(self, frame):
        self.frames.append(frame)


class ResettableProcessor:
    def reset(self):
        return None


def test_record_loop_sends_stop_action_after_chunk_and_episode_end(monkeypatch):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }
    robot = RecordingRobot(events)
    policy = ChunkPolicy()
    dataset = DummyDataset()
    predicted_actions = [
        {
            "arm_shoulder_pan.pos": 1.0,
            "x.vel": 0.3,
            "y.vel": -0.1,
            "theta.vel": 20.0,
        },
        {
            "arm_shoulder_pan.pos": 2.0,
            "x.vel": 0.4,
            "y.vel": 0.2,
            "theta.vel": 35.0,
        },
    ]

    def fake_predict_action(**kwargs):
        kwargs["policy"]._action_queue.popleft()
        return predicted_actions.pop(0)

    monkeypatch.setattr(lerobot_record, "predict_action", fake_predict_action)
    monkeypatch.setattr(lerobot_record, "make_robot_action", lambda action_values, features: action_values)
    monkeypatch.setattr(
        lerobot_record,
        "build_dataset_frame",
        lambda features, values, prefix=None: dict(values),
    )
    monkeypatch.setattr(lerobot_record, "precise_sleep", lambda _: None)

    lerobot_record.record_loop(
        robot=robot,
        events=events,
        fps=30,
        teleop_action_processor=lambda value: value[0],
        robot_action_processor=lambda value: value[0],
        robot_observation_processor=lambda obs: obs,
        dataset=dataset,
        policy=policy,
        preprocessor=ResettableProcessor(),
        postprocessor=ResettableProcessor(),
        control_time_s=10,
        single_task="test task",
        stop_action_builder=lambda action, obs: {
            **action,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        },
    )

    assert robot.sent_actions == [
        {
            "arm_shoulder_pan.pos": 1.0,
            "x.vel": 0.3,
            "y.vel": -0.1,
            "theta.vel": 20.0,
        },
        {
            "arm_shoulder_pan.pos": 2.0,
            "x.vel": 0.4,
            "y.vel": 0.2,
            "theta.vel": 35.0,
        },
        {
            "arm_shoulder_pan.pos": 2.0,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        },
        {
            "arm_shoulder_pan.pos": 2.0,
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
        },
    ]
    assert len(dataset.frames) == 2
    assert [frame["x.vel"] for frame in dataset.frames] == [0.3, 0.4]


class DummySocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send_string(self, payload):
        self.sent.append(payload)

    def close(self):
        self.closed = True


class DummyContext:
    def __init__(self):
        self.terminated = False

    def term(self):
        self.terminated = True


def test_lekiwi_client_disconnect_sends_zero_base_command():
    client = LeKiwiClient.__new__(LeKiwiClient)
    client.zmq_cmd_socket = DummySocket()
    client.zmq_observation_socket = DummySocket()
    client.zmq_context = DummyContext()
    client._is_connected = True

    client.disconnect()

    assert json.loads(client.zmq_cmd_socket.sent[0]) == {
        "x.vel": 0.0,
        "y.vel": 0.0,
        "theta.vel": 0.0,
    }
    assert client.zmq_cmd_socket.closed is True
    assert client.zmq_observation_socket.closed is True
    assert client.zmq_context.terminated is True
    assert client._is_connected is False
