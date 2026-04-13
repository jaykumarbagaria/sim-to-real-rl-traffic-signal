"""
traffic_signal_env.py
=====================
Complete RL pipeline for "Sim-to-Real Gap in Traffic Signal Control"

Contains
--------
  CalibParams      : read-only calibrated parameters (DO NOT MODIFY)
  RouteGenerator   : ENV A and ENV B route file generation
  SumoFileManager  : idempotent SUMO file creation and validation
  inspect_network  : discovers real TLS/lane IDs from a live simulation
  TrafficSignalEnv : RL environment wrapper (TraCI, no Gym)
  ReplayBuffer     : experience replay for DQN
  DQN              : two-layer PyTorch Q-network (Double DQN)
  DQNAgent         : epsilon-greedy agent
  train_agent      : training loop with per-episode ENV B resampling
  evaluate_agent   : greedy evaluation
  cross_evaluate   : 2x2 train/test matrix
  plot_*           : paper-ready figures
  print_results_table : clean console table

Scientific constraints (FIXED — never modify)
----------------------------------------------
  ENV A: constant headway=2.0s, startup=1.0s, deterministic
  ENV B: Lognormal(mu=0.8628, sigma=0.5416) headway,
         Normal(mu=2.5661, std=1.7738) startup,
         position model h_i = 2.38 + 0.57*exp(-0.07*i)
"""

import os
import sys
import time
import subprocess
import collections
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# TraCI bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _bootstrap_traci():
    sumo_home = os.environ.get('SUMO_HOME', '/usr/share/sumo')
    os.environ['SUMO_HOME'] = sumo_home
    tools = os.path.join(sumo_home, 'tools')
    if tools not in sys.path:
        sys.path.insert(0, tools)
    import traci
    return traci


def _kill_sumo():
    subprocess.run(['killall', '-9', 'sumo'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Calibrated parameters  — READ-ONLY
# ─────────────────────────────────────────────────────────────────────────────

class CalibParams:
    """
    Ground-truth calibration from 11 Hungarian/Polish intersections.
    Values locked — see calibration notebook for derivation.
    """
    # Lognormal headway distribution
    HW_MU     = 0.8628
    HW_SIGMA  = 0.5416

    # Normal startup delay distribution
    SD_MU     = 2.5661
    SD_STD    = 1.7738

    # Position-based discharge  h_i = POS_A + POS_B * exp(-POS_C * i)
    POS_A     = 2.38
    POS_B     = 0.57
    POS_C     = 0.07

    # ENV A constants
    ENVA_HEADWAY = 2.0
    ENVA_STARTUP = 1.0

    # Physical floor
    MIN_HEADWAY  = 0.5

    @staticmethod
    def pos_headway(i: int) -> float:
        """h_i = a + b*exp(-c*i),  i is 1-indexed queue position."""
        return (CalibParams.POS_A
                + CalibParams.POS_B * np.exp(-CalibParams.POS_C * float(i)))

    @staticmethod
    def lognorm_mean() -> float:
        return float(np.exp(CalibParams.HW_MU + CalibParams.HW_SIGMA ** 2 / 2))


# ─────────────────────────────────────────────────────────────────────────────
# Route generation
# ─────────────────────────────────────────────────────────────────────────────

class RouteGenerator:
    """
    Generates SUMO .rou.xml files for ENV A and ENV B.

    ENV A — deterministic
      Vehicles depart at fixed 2.0 s intervals; first vehicle gets +1.0 s startup.

    ENV B — stochastic, calibrated
      h_i = (a + b*exp(-c*i)) + (Lognormal(mu,sigma) - E[Lognormal])
      First vehicle additionally gets startup drawn from Normal(mu_sd, std_sd).
      Clipped to MIN_HEADWAY physical floor.
    """

    _VTYPE = """\
<vType id="car"
    accel="2.6" decel="4.5" sigma="0.3"
    tau="0.6" minGap="1.5" maxSpeed="13.9"/>"""

    _ROUTE = """ 
<route id="r0" edges="A0A1 A1B1"/>
<route id="r1" edges="B0B1 B1B0"/>
"""

    def __init__(self, n_vehicles: int = 60, start_time: float = 10.0):
        self.n    = n_vehicles
        self.t0   = start_time

    def _to_xml(self, headways: np.ndarray) -> str:
        deps  = self.t0 + np.cumsum(headways)
        lines = ['<routes>', self._VTYPE, self._ROUTE]
        for i, t in enumerate(deps):
          route_id = "r0" if i % 2 == 0 else "r1"
          lines.append(f'<vehicle id="veh{i}" type="car" route="{route_id}" depart="{t:.2f}" departSpeed="max"/>')
        lines.append('</routes>')
        return '\n'.join(lines)

    # ── ENV A ──────────────────────────────────────────────────────────────

    def env_a_headways(self) -> np.ndarray:
        p = CalibParams
        h    = np.full(self.n, p.ENVA_HEADWAY, dtype=float)
        h[0] = p.ENVA_HEADWAY + p.ENVA_STARTUP
        return h

    def write_env_a(self, path: str = 'routes_envA.rou.xml') -> str:
        with open(path, 'w') as f:
            f.write(self._to_xml(self.env_a_headways()))
        return path

    # ── ENV B ──────────────────────────────────────────────────────────────

    def env_b_headways(self) -> np.ndarray:
        """
        Hybrid: position-based deterministic + zero-meaned lognormal noise.
        Each call produces a fresh stochastic sample (call once per episode).
        """
        p   = CalibParams
        idx = np.arange(1, self.n + 1, dtype=float)

        h_det  = p.POS_A + p.POS_B * np.exp(-p.POS_C * idx)
        lm     = p.lognorm_mean()
        noise  = np.random.lognormal(p.HW_MU, p.HW_SIGMA, self.n) - lm
        h      = h_det + noise

        # Startup delay for first vehicle
        sd   = max(np.random.normal(p.SD_MU, p.SD_STD), 0.5)
        h[0] = max(h[0] + sd, p.MIN_HEADWAY)

        return np.clip(h, p.MIN_HEADWAY, None)

    def write_env_b(self, path: str = 'routes_envB.rou.xml') -> str:
        with open(path, 'w') as f:
            f.write(self._to_xml(self.env_b_headways()))
        return path


# ─────────────────────────────────────────────────────────────────────────────
# SUMO file manager
# ─────────────────────────────────────────────────────────────────────────────

class SumoFileManager:
    """
    Creates and validates all SUMO files. All operations are idempotent.

    File layout
    -----------
    net.net.xml           shared network (hand-written, TLS at B1 guaranteed)
    additional.add.xml    empty additional file
    routes_envA.rou.xml   ENV A routes (deterministic)
    routes_envB.rou.xml   ENV B routes (regenerated each episode)
    config_envA.sumocfg   ENV A config
    config_envB.sumocfg   ENV B config
    """

    NET   = 'net.net.xml'
    ADD   = 'additional.add.xml'
    RA    = 'routes_envA.rou.xml'
    RB    = 'routes_envB.rou.xml'
    CA    = 'config_envA.sumocfg'
    CB    = 'config_envB.sumocfg'
    T_END = 1000

    def __init__(self, n_vehicles: int = 60):
        self.rgen = RouteGenerator(n_vehicles=n_vehicles)

    def build_all(self):
        """Call once per Colab session to create all files."""
        print("Building SUMO files ...")
        self._net()
        self._additional()
        self.rgen.write_env_a(self.RA)
        self.rgen.write_env_b(self.RB)
        self._config(self.RA, self.CA)
        self._config(self.RB, self.CB)
        self._validate(self.CA, 'ENV A')
        self._validate(self.CB, 'ENV B')
        print("All SUMO files ready.\n")

    def rebuild_routes_only(self):
        """Regenerate stochastic ENV B routes. Call at start of each episode."""
        self.rgen.write_env_b(self.RB)

    def _net(self):
        """
        Write net.net.xml by hand — no netgenerate, no netconvert, no subprocess.

        Topology
        --------
        One signalised junction B1 with two approaches and two exits.

          A0 --[A0A1]--> B1 --[A1B1]--> A1   (west-east through road)
          B0 --[B0B1]--> B1 --[B1B0]--> B1_out  (south-north through road)

        TLS ID  : B1
        Lane IDs: A0A1_0, B0B1_0, A1B1_0, B1B0_0
        """
        net_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<net version="1.16" junctionCornerDetail="5"\n'
            '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n'
            '     xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">\n'
            '\n'
            '    <location netOffset="0.00,0.00"\n'
            '              convBoundary="-200.00,-200.00,200.00,200.00"\n'
            '              origBoundary="-200.00,-200.00,200.00,200.00"\n'
            '              projParameter="!"/>\n'
            '\n'
            '    <!-- Edges -->\n'
            '    <edge id="A0A1" from="A0" to="B1" priority="2">\n'
            '        <lane id="A0A1_0" index="0" disallow="pedestrian"\n'
            '              speed="13.89" length="196.80"\n'
            '              shape="-200.00,1.60 -3.20,1.60"/>\n'
            '    </edge>\n'
            '\n'
            '    <edge id="A1B1" from="B1" to="A1" priority="2">\n'
            '        <lane id="A1B1_0" index="0" disallow="pedestrian"\n'
            '              speed="13.89" length="196.80"\n'
            '              shape="3.20,-1.60 200.00,-1.60"/>\n'
            '    </edge>\n'
            '\n'
            '    <edge id="B0B1" from="B0" to="B1" priority="2">\n'
            '        <lane id="B0B1_0" index="0" disallow="pedestrian"\n'
            '              speed="13.89" length="196.80"\n'
            '              shape="1.60,-200.00 1.60,-3.20"/>\n'
            '    </edge>\n'
            '\n'
            '    <edge id="B1B0" from="B1" to="B1_out" priority="2">\n'
            '        <lane id="B1B0_0" index="0" disallow="pedestrian"\n'
            '              speed="13.89" length="196.80"\n'
            '              shape="-1.60,3.20 -1.60,200.00"/>\n'
            '    </edge>\n'
            '\n'
            '    <!-- Internal lanes -->\n'
            '    <edge id=":B1_0" function="internal">\n'
            '        <lane id=":B1_0_0" index="0" disallow="pedestrian"\n'
            '              speed="8.57" length="6.40"\n'
            '              shape="-1.60,1.60 1.60,-1.60"/>\n'
            '    </edge>\n'
            '    <edge id=":B1_1" function="internal">\n'
            '        <lane id=":B1_1_0" index="0" disallow="pedestrian"\n'
            '              speed="8.57" length="6.40"\n'
            '              shape="1.60,-1.60 -1.60,1.60"/>\n'
            '    </edge>\n'
            '\n'
            '    <!-- Traffic-light logic -->\n'
            '    <tlLogic id="B1" type="static" programID="0" offset="0">\n'
            '        <phase duration="38" state="Gr"/>\n'
            '        <phase duration="3"  state="yr"/>\n'
            '        <phase duration="38" state="rG"/>\n'
            '        <phase duration="3"  state="ry"/>\n'
            '    </tlLogic>\n'
            '\n'
            '    <!-- Junctions -->\n'
            '    <junction id="B1" type="traffic_light"\n'
            '              x="0.00" y="0.00"\n'
            '              incLanes="A0A1_0 B0B1_0"\n'
            '              intLanes=":B1_0_0 :B1_1_0"\n'
            '              shape="-3.20,3.20 3.20,3.20 3.20,-3.20 -3.20,-3.20">\n'
            '        <request index="0" response="10" foes="10" cont="0"/>\n'
            '        <request index="1" response="01" foes="01" cont="0"/>\n'
            '    </junction>\n'
            '\n'
            '    <junction id="A0" type="dead_end" x="-200.00" y="0.00"\n'
            '              incLanes="" intLanes=""\n'
            '              shape="-200.00,-1.60 -200.00,1.60"/>\n'
            '\n'
            '    <junction id="A1" type="dead_end" x="200.00" y="0.00"\n'
            '              incLanes="A1B1_0" intLanes=""\n'
            '              shape="200.00,1.60 200.00,-1.60"/>\n'
            '\n'
            '    <junction id="B0" type="dead_end" x="0.00" y="-200.00"\n'
            '              incLanes="" intLanes=""\n'
            '              shape="1.60,-200.00 -1.60,-200.00"/>\n'
            '\n'
            '    <junction id="B1_out" type="dead_end" x="0.00" y="200.00"\n'
            '              incLanes="B1B0_0" intLanes=""\n'
            '              shape="-1.60,200.00 1.60,200.00"/>\n'
            '\n'
            '    <!-- Connections -->\n'
            '    <connection from="A0A1" to="A1B1" fromLane="0" toLane="0"\n'
            '                via=":B1_0_0" tl="B1" linkIndex="0" dir="s" state="o"/>\n'
            '    <connection from="B0B1" to="B1B0" fromLane="0" toLane="0"\n'
            '                via=":B1_1_0" tl="B1" linkIndex="1" dir="s" state="o"/>\n'
            '    <connection from=":B1_0" to="A1B1" fromLane="0" toLane="0"\n'
            '                dir="s" state="M"/>\n'
            '    <connection from=":B1_1" to="B1B0" fromLane="0" toLane="0"\n'
            '                dir="s" state="M"/>\n'
            '\n'
            '</net>\n'
        )
        with open(self.NET, 'w') as f:
            f.write(net_xml)
        print(f"  {self.NET} written (hand-crafted, TLS B1 guaranteed)")

    def _additional(self):
        with open(self.ADD, 'w') as f:
            f.write('<additional>\n</additional>\n')

    def _config(self, route_file: str, cfg_path: str):
        xml = (
            '<configuration>\n'
            '  <input>\n'
            f'    <net-file value="{self.NET}"/>\n'
            f'    <route-files value="{route_file}"/>\n'
            f'    <additional-files value="{self.ADD}"/>\n'
            '  </input>\n'
            '  <time>\n'
            '    <begin value="0"/>\n'
            f'    <end value="{self.T_END}"/>\n'
            '  </time>\n'
            '</configuration>\n'
        )
        with open(cfg_path, 'w') as f:
            f.write(xml)

    def _validate(self, cfg: str, label: str):
        r = subprocess.run(
            ['sumo', '-c', cfg, '--quit-on-end', '--no-warnings', 'true'],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"Config invalid [{label}]:\n{r.stderr[:400]}")
        print(f"  {cfg}  [{label}] ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Network inspector
# ─────────────────────────────────────────────────────────────────────────────

def inspect_network(config_path: str, step_length: float = 0.1) -> dict:
    """
    Start SUMO for one step, print TLS and lane IDs, close.
    Run once after build_all() to get IDs for TrafficSignalEnv.
    """
    traci = _bootstrap_traci()
    _kill_sumo()

    traci.start(
        ['sumo', '-c', config_path,
         '--step-length', str(step_length),
         '--no-warnings', 'true', '--start'],
        label='inspector'
    )
    traci.simulationStep()

    tls_ids  = list(traci.trafficlight.getIDList())
    lane_ids = [l for l in traci.lane.getIDList() if not l.startswith(':')]

    traci.close()
    _kill_sumo()

    print("=== Traffic Light IDs ===")
    for tid in tls_ids:
        print(f"  '{tid}'")

    print("\n=== Driveable Lane IDs ===")
    for lid in lane_ids:
        print(f"  '{lid}'")

    print(f"\n-> Use tls_id='{tls_ids[0] if tls_ids else 'B1'}'"
          f" and a subset of the lane IDs above.")

    return {'tls_ids': tls_ids, 'lane_ids': lane_ids}


# ─────────────────────────────────────────────────────────────────────────────
# RL Environment
# ─────────────────────────────────────────────────────────────────────────────

class TrafficSignalEnv:
    """
    Minimal RL environment wrapper (TraCI, no Gym).

    State — shape (4 * n_lanes,),  float32,  normalised
    -------
      Per lane: [queue_length/20, waiting_time/300,
                 time_since_green/120, discharge_rate/1.0]
        discharge_rate = 1 / pos_headway(queue_length)  in veh/s

    Action — int {0=hold, 1=switch phase}
    -------

    Reward
    ------
      R = -(delay + 0.5*stops + 0.1*phase_switch)
      Same formula for both ENV A and ENV B — scientific requirement.
    """

    NFEATURES = 4
    NORM = np.array([20.0, 300.0, 120.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        config_path: str,
        tls_id: str,
        lane_ids: list,
        step_length: float = 0.1,
        max_steps: int     = 2000,
        use_gui: bool      = False,
        label: str         = 'env',
    ):
        self.config_path = config_path
        self.tls_id      = tls_id
        self.lane_ids    = lane_ids
        self.n_lanes     = len(lane_ids)
        self.step_length = step_length
        self.max_steps   = max_steps
        self.use_gui     = use_gui
        self.label       = label

        self.traci     = _bootstrap_traci()
        self.state_dim = self.NFEATURES * self.n_lanes

        self._step      = 0
        self._green_t   = 0.0
        self._cur_phase = 0
        self._n_sw      = 0

    def reset(self) -> np.ndarray:
        self._safe_close()
        binary = 'sumo-gui' if self.use_gui else 'sumo'
        self.traci.start(
            [binary, '-c', self.config_path,
             '--step-length', str(self.step_length),
             '--no-warnings', 'true', '--start'],
            label=self.label
        )
        self._step    = 0
        self._n_sw    = 0
        self._green_t = self.traci.simulation.getTime()
        self._cur_phase = self.traci.trafficlight.getPhase(self.tls_id)
        return self._state()

    def step(self, action: int):
        switched = self._act(action)
        self.traci.simulationStep()
        self._step += 1

        s = self._state()
        r = self._reward(switched)
        done = (
            self._step >= self.max_steps
            or self.traci.simulation.getMinExpectedNumber() == 0
        )
        info = {
            'step':    self._step,
            'switches': self._n_sw,
            'delay':   sum(self.traci.lane.getWaitingTime(l) for l in self.lane_ids),
            'stops':   sum(self.traci.lane.getLastStepHaltingNumber(l) for l in self.lane_ids),
        }
        return s, r, done, info

    def close(self):
        self._safe_close()

    # ── private ─────────────────────────────────────────────────────────────

    def _act(self, action: int) -> bool:
        if action == 1:
            logics  = self.traci.trafficlight.getAllProgramLogics(self.tls_id)
            n_ph    = len(logics[0].phases)
            next_ph = (self._cur_phase + 1) % n_ph
            self.traci.trafficlight.setPhase(self.tls_id, next_ph)
            self._cur_phase = next_ph
            self._green_t   = self.traci.simulation.getTime()
            self._n_sw     += 1
            return True
        return False

    def _state(self) -> np.ndarray:
        tsg = self.traci.simulation.getTime() - self._green_t
        feats = []
        for lid in self.lane_ids:
            q    = float(self.traci.lane.getLastStepHaltingNumber(lid))
            w    = float(self.traci.lane.getWaitingTime(lid))
            pos  = max(1, int(q))
            hi   = CalibParams.pos_headway(pos)
            rate = 1.0 / hi if hi > 0 else 0.5
            raw  = np.array([q, w, tsg, rate], dtype=np.float32)
            feats.append(raw / self.NORM)
        return np.concatenate(feats, dtype=np.float32)

    def _reward(self, switched: bool) -> float:
        delay  = sum(self.traci.lane.getWaitingTime(l)            for l in self.lane_ids)
        stops  = sum(self.traci.lane.getLastStepHaltingNumber(l)  for l in self.lane_ids)
        return float(-(delay + 0.5 * stops + 0.1 * float(switched)))

    def _safe_close(self):
        try:
            if self.traci.isLoaded():
                self.traci.close()
        except Exception:
            pass
        _kill_sumo()


# ─────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buf.append((
            np.array(s,  dtype=np.float32), int(a), float(r),
            np.array(ns, dtype=np.float32), bool(done)
        ))

    def sample(self, n: int):
        batch = random.sample(self.buf, n)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────────────────────────
# DQN
# ─────────────────────────────────────────────────────────────────────────────

def _torch():
    try:
        import torch, torch.nn as nn, torch.optim as optim
        return torch, nn, optim
    except ImportError:
        raise ImportError("Install PyTorch: !pip install torch")


class DQN:
    """
    Two hidden-layer Q-network (128 units each).
    Uses Double DQN for target computation.
    """

    def __init__(self, state_dim: int, lr: float = 1e-3):
        torch, nn, optim = _torch()
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        def _make_net():
            return nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128),       nn.ReLU(),
                nn.Linear(128, 2),
            ).to(self.dev)

        self.online = _make_net()
        self.target = _make_net()
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self._t  = torch
        self._nn = nn

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with self._t.no_grad():
            s = self._t.FloatTensor(state).unsqueeze(0).to(self.dev)
            return self.online(s).cpu().numpy()[0]

    def train_step(self, buf: ReplayBuffer, batch: int = 64, gamma: float = 0.99) -> float:
        if len(buf) < batch:
            return 0.0
        torch, nn, _ = _torch()
        s, a, r, ns, d = buf.sample(batch)
        S  = torch.FloatTensor(s).to(self.dev)
        A  = torch.LongTensor(a).to(self.dev)
        R  = torch.FloatTensor(r).to(self.dev)
        NS = torch.FloatTensor(ns).to(self.dev)
        D  = torch.FloatTensor(d).to(self.dev)

        q     = self.online(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_best = self.online(NS).argmax(1)
            q_next = self.target(NS).gather(1, a_best.unsqueeze(1)).squeeze(1)
            q_tgt  = R + gamma * q_next * (1.0 - D)

        loss = nn.functional.smooth_l1_loss(q, q_tgt)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str):
        torch, _, _ = _torch()
        torch.save(self.online.state_dict(), path)
        print(f"  Saved model: {path}")

    def load(self, path: str):
        torch, _, _ = _torch()
        self.online.load_state_dict(
            torch.load(path, map_location=self.dev))
        self.sync_target()


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """Epsilon-greedy agent wrapping DQN + ReplayBuffer."""

    def __init__(
        self,
        state_dim:   int,
        eps_start:   float = 1.0,
        eps_end:     float = 0.05,
        eps_decay:   int   = 50_000,
        lr:          float = 1e-3,
        target_sync: int   = 500,
        batch_size:  int   = 64,
        gamma:       float = 0.99,
        buf_size:    int   = 20_000,
    ):
        self.dqn         = DQN(state_dim, lr=lr)
        self.buf         = ReplayBuffer(buf_size)
        self.eps         = eps_start
        self.eps_end     = eps_end
        self.eps_decay   = eps_decay
        self.target_sync = target_sync
        self.batch_size  = batch_size
        self.gamma       = gamma
        self._steps      = 0

    def act(self, state: np.ndarray) -> int:
        if random.random() < self.eps:
            return random.randint(0, 1)
        return int(np.argmax(self.dqn.q_values(state)))

    def push(self, s, a, r, ns, done):
        self.buf.push(s, a, r, ns, done)

    def learn(self) -> float:
        loss = self.dqn.train_step(self.buf, self.batch_size, self.gamma)
        self._steps += 1
        frac     = min(1.0, self._steps / self.eps_decay)
        self.eps = self.eps_end + (1.0 - self.eps_end) * (1.0 - frac)
        if self._steps % self.target_sync == 0:
            self.dqn.sync_target()
        return loss


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_agent(
    env:           TrafficSignalEnv,
    agent:         DQNAgent,
    n_episodes:    int = 300,
    file_manager:  SumoFileManager = None,
    env_label:     str = '',
    verbose_every: int = 25,
) -> dict:
    """
    Training loop.

    file_manager: if provided (ENV B only), ENV B routes are regenerated each
    episode so the agent sees a fresh stochastic discharge sample per episode.
    This is essential for generalisation — without it the agent overfits to one
    fixed route file.
    """
    hist = {k: [] for k in
            ['ep_reward', 'ep_delay', 'ep_stops', 'ep_steps', 'loss', 'eps']}

    print(f"\n{'='*60}")
    print(f"Training  |  env={env_label}  |  n_episodes={n_episodes}")
    print(f"{'='*60}")

    for ep in range(1, n_episodes + 1):
        if file_manager is not None:
            file_manager.rebuild_routes_only()

        state = env.reset()
        ep_r = ep_d = ep_s = 0.0
        losses = []

        while True:
            a                       = agent.act(state)
            ns, reward, done, info  = env.step(a)
            agent.push(state, a, reward, ns, done)
            l = agent.learn()
            if l > 0:
                losses.append(l)
            ep_r += reward
            ep_d += info['delay']
            ep_s += info['stops']
            state = ns
            if done:
                break

        n = max(info['step'], 1)
        hist['ep_reward'].append(ep_r)
        hist['ep_delay'].append(ep_d / n)
        hist['ep_stops'].append(ep_s / n)
        hist['ep_steps'].append(info['step'])
        hist['loss'].append(float(np.mean(losses)) if losses else 0.0)
        hist['eps'].append(agent.eps)

        if ep % verbose_every == 0 or ep == 1:
            avg = np.mean(hist['ep_reward'][-verbose_every:])
            print(
                f"  ep {ep:4d}/{n_episodes}  "
                f"reward={ep_r:8.1f}  avg={avg:8.1f}  "
                f"delay/step={hist['ep_delay'][-1]:5.2f}  "
                f"eps={agent.eps:.3f}  loss={hist['loss'][-1]:.4f}"
            )

    env.close()
    print(f"Done — {n_episodes} episodes.\n")
    return hist


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    env:          TrafficSignalEnv,
    agent:        DQNAgent,
    n_episodes:   int = 20,
    file_manager: SumoFileManager = None,
    label:        str = '',
) -> dict:
    """Greedy evaluation (eps=0). Returns metric dict."""
    saved = agent.eps
    agent.eps = 0.0

    rewards, delays, stops = [], [], []

    for _ in range(n_episodes):
        if file_manager is not None:
            file_manager.rebuild_routes_only()

        state = env.reset()
        ep_r = ep_d = ep_s = 0.0

        while True:
            a              = agent.act(state)
            state, r, done, info = env.step(a)
            ep_r += r
            ep_d += info['delay']
            ep_s += info['stops']
            if done:
                break

        n = max(info['step'], 1)
        rewards.append(ep_r)
        delays.append(ep_d / n)
        stops.append(ep_s / n)

    agent.eps = saved
    env.close()

    res = {
        'label':       label,
        'mean_reward': float(np.mean(rewards)),
        'std_reward':  float(np.std(rewards)),
        'mean_delay':  float(np.mean(delays)),
        'mean_stops':  float(np.mean(stops)),
        'all_rewards': rewards,
        'all_delays':  delays,
    }
    print(
        f"  [{label:<30}]  "
        f"reward={res['mean_reward']:8.1f}±{res['std_reward']:.1f}  "
        f"delay={res['mean_delay']:5.2f}  stops={res['mean_stops']:4.2f}"
    )
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Cross-evaluation
# ─────────────────────────────────────────────────────────────────────────────

def cross_evaluate(
    agent_a: DQNAgent, agent_b: DQNAgent,
    env_a: TrafficSignalEnv, env_b: TrafficSignalEnv,
    n_episodes: int = 20,
    fm_b: SumoFileManager = None,
) -> dict:
    """
    2x2 train/test matrix.

    A->A: baseline
    A->B: sim-to-real gap (core result)
    B->B: oracle
    B->A: reverse transfer
    """
    print(f"\n{'='*60}")
    print("CROSS EVALUATION")
    print(f"{'='*60}")

    mx = {}
    mx['A_on_A'] = evaluate_agent(env_a, agent_a, n_episodes,        label='Train=A  Test=A')
    mx['A_on_B'] = evaluate_agent(env_b, agent_a, n_episodes, fm_b,  label='Train=A  Test=B  <- gap')
    mx['B_on_B'] = evaluate_agent(env_b, agent_b, n_episodes, fm_b,  label='Train=B  Test=B')
    mx['B_on_A'] = evaluate_agent(env_a, agent_b, n_episodes,        label='Train=B  Test=A')

    r_aa = mx['A_on_A']['mean_reward']
    r_ab = mx['A_on_B']['mean_reward']
    r_bb = mx['B_on_B']['mean_reward']
    gap  = r_ab - r_aa

    mx.update({'gap': gap, 'r_aa': r_aa, 'r_ab': r_ab, 'r_bb': r_bb})

    pct = gap / abs(r_aa) * 100 if r_aa != 0 else 0
    print(f"\n  Sim-to-real gap:  R(A->A)={r_aa:.1f}  R(A->B)={r_ab:.1f}  "
          f"gap={gap:.1f} ({pct:.1f}%)")
    print(f"  Oracle vs gap:    R(B->B)={r_bb:.1f}  improvement={r_bb-r_ab:.1f}")
    return mx


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(v, w=20):
    return [np.mean(v[max(0, i-w):i+1]) for i in range(len(v))]


def plot_learning_curves(hist_a: dict, hist_b: dict,
                          path: str = 'learning_curves.png'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    specs = [('ep_reward', 'Episode Reward'),
             ('ep_delay',  'Mean Delay / Step (s)'),
             ('loss',      'TD Loss')]

    for ax, (key, ylabel) in zip(axes, specs):
        ax.plot(_smooth(hist_a[key]), color='green',   lw=2, label='Agent A (ENV A)')
        ax.plot(_smooth(hist_b[key]), color='crimson', lw=2, label='Agent B (ENV B)')
        ax.set_xlabel('Episode', fontsize=11)
        ax.set_ylabel(ylabel,    fontsize=11)
        ax.set_title(ylabel,     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.suptitle('DQN Training Curves', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_cross_evaluation(mx: dict, path: str = 'cross_evaluation.png'):
    keys    = ['A_on_A', 'A_on_B', 'B_on_B', 'B_on_A']
    labels  = ['Train=A\nTest=A', 'Train=A\nTest=B\n(gap)',
               'Train=B\nTest=B', 'Train=B\nTest=A']
    colors  = ['green', 'crimson', 'steelblue', 'orange']
    metrics = [('mean_reward', 'Mean Episode Reward'),
               ('mean_delay',  'Mean Delay / Step (s)'),
               ('mean_stops',  'Mean Stops / Step')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel) in zip(axes, metrics):
        vals = [mx[k][metric] for k in keys]
        errs = ([mx[k]['std_reward'] for k in keys]
                if metric == 'mean_reward' else None)
        bars = ax.bar(labels, vals, color=colors, alpha=0.85,
                      edgecolor='white', width=0.6,
                      yerr=errs, capsize=4 if errs else 0)
        for bar, val in zip(bars, vals):
            ypos = bar.get_height() + abs(bar.get_height()) * 0.02
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel,  fontsize=11)
        ax.grid(axis='y', alpha=0.3)

    r_aa, r_ab = mx['r_aa'], mx['r_ab']
    axes[0].annotate(
        f"Gap\n{r_ab - r_aa:.1f}",
        xy=(1, r_ab), xytext=(1.6, (r_aa + r_ab) / 2),
        fontsize=9, color='crimson',
        arrowprops=dict(arrowstyle='->', color='crimson'),
    )

    plt.suptitle('Sim-to-Real Gap: Cross-Evaluation Results', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_reward_distributions(mx: dict, path: str = 'reward_distributions.png'):
    keys   = ['A_on_A', 'A_on_B', 'B_on_B', 'B_on_A']
    labels = ['Train=A\nTest=A', 'Train=A\nTest=B',
              'Train=B\nTest=B', 'Train=B\nTest=A']
    colors = ['green', 'crimson', 'steelblue', 'orange']

    fig, ax = plt.subplots(figsize=(10, 5))
    data = [mx[k]['all_rewards'] for k in keys]

    vp = ax.violinplot(data, positions=range(4),
                       showmeans=True, showmedians=True)
    for pc, col in zip(vp['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.75)

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Reward Distribution by Train/Test Condition', fontsize=13)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def print_results_table(mx: dict):
    keys   = ['A_on_A', 'A_on_B', 'B_on_B', 'B_on_A']
    labels = ['Train=A  Test=A (baseline)',
              'Train=A  Test=B (sim-to-real gap)',
              'Train=B  Test=B (oracle)',
              'Train=B  Test=A (reverse transfer)']
    print("\n" + "="*72)
    print("RESULTS TABLE")
    print("="*72)
    print(f"{'Condition':<40}  {'Reward':>8}  {'Delay':>8}  {'Stops':>8}")
    print("-"*72)
    for k, lbl in zip(keys, labels):
        r = mx[k]
        print(f"{lbl:<40}  {r['mean_reward']:>8.2f}  "
              f"{r['mean_delay']:>8.3f}  {r['mean_stops']:>8.3f}")
    print("="*72)
    pct = mx['gap'] / abs(mx['r_aa']) * 100 if mx['r_aa'] != 0 else 0
    print(f"\nSim-to-real degradation: {mx['gap']:.2f} ({pct:.1f}%)")
    print(f"Oracle improvement over gap: {mx['r_bb'] - mx['r_ab']:.2f}")
