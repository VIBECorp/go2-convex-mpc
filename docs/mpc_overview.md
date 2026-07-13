# Convex MPC 로코모션 컨트롤러 설명

이 문서는 본 저장소에 구현된 **접촉력 기반 Convex Model Predictive Control (MPC)** 의 수식과
코드 구조를 설명한다. 구현은 아래 논문의 방법론을 따른다.

> *"Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"* (Di Carlo et al., IROS 2018)
> https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf

지원 로봇: **Unitree Go2** (~15 kg), **Unitree A2** (~40 kg).

---

## 1. 전체 제어 아키텍처

```
명령 (x/y 속도, x/y 위치, 높이, roll/pitch(+rate), yaw rate)
        │
        ▼
┌──────────────────────────┐   ~48 Hz (게이트 주기/16)
│ 기준 궤적 생성기          │   com_trajectory.py :: ComTraj.generate_traj
│  - COM 위치/자세/속도 궤적 │
│  - 접촉 테이블            │
│  - 발 lever arm 예측      │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐   ~48 Hz
│ Centroidal MPC (QP)      │   centroidal_mpc.py :: CentroidalMPC.solve_QP
│  - OSQP (CasADi conic)   │
│  → 4발 접촉력 f ∈ R^12   │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐   200 Hz
│ 다리 제어기               │   leg_controller.py :: LegController
│  - stance: τ = Jᵀ(−f)+보상│
│  - swing : 임피던스+FF    │
└──────────┬───────────────┘
           ▼
     MuJoCo 시뮬레이션 (1000 Hz) / 실로봇
```

| 모듈 | 주기 | 파일 |
|---|---|---|
| MuJoCo 물리 시뮬레이션 | 1000 Hz | `mujoco_model.py` |
| 다리 제어기 (토크 계산) | 200 Hz | `leg_controller.py` |
| MPC + 기준 궤적 갱신 | ~48 Hz (`MPC_DT = 게이트주기/16`) | `centroidal_mpc.py`, `com_trajectory.py` |
| 게이트 스케줄러 | 매 제어 틱에서 평가 | `gait.py` |
| 로봇 동역학 (Pinocchio) | 매 제어 틱 갱신 | `go2_robot_data.py` |

MPC 예측 지평은 **게이트 한 주기**(trot 3 Hz → 0.333 s)를 **N = 16 스텝**으로 나눈다.

---

## 2. 단순화 강체 (Centroidal) 동역학 모델

로봇 전체를 COM에 위치한 **하나의 강체**로 근사한다 (다리 질량/관성은 관성텐서에 포함되지만
다리 운동에 의한 동적 효과는 무시). 상태는 12차원:

$$
x = [\,p_x,\ p_y,\ p_z,\ \phi,\ \theta,\ \psi,\ v_x,\ v_y,\ v_z,\ \omega_x,\ \omega_y,\ \omega_z\,]^T
$$

- $p$: COM 위치 (월드), $v$: COM 속도 (월드)
- $(\phi,\theta,\psi)$: roll/pitch/yaw (ZYX 오일러, 월드 기준, yaw는 unwrap 처리)
- $\omega$: 각속도 (월드)

입력은 4발의 접촉력 (월드): $u = [f_{FL}; f_{FR}; f_{RL}; f_{RR}] \in \mathbb{R}^{12}$.

강체 운동방정식:

$$
\dot p = v, \qquad
\dot v = \frac{1}{m}\sum_i f_i + g, \qquad
\dot\omega \approx I^{-1}\sum_i r_i \times f_i, \qquad
\begin{bmatrix}\dot\phi\\\dot\theta\\\dot\psi\end{bmatrix} \approx R_z(\psi)^T\,\omega
$$

핵심 근사 (문제를 **볼록(convex)** 하게 만드는 가정):

1. **Yaw만 선형화**: 자세 운동학을 $R_z(\bar\psi)^T$로 근사 (roll/pitch가 작다고 가정,
   $\bar\psi$는 지평 평균 yaw). → roll/pitch 명령은 ±0.25 rad 이내 권장.
2. **관성텐서 고정**: $I$는 solve 시점의 월드 프레임 centroidal 관성 (`pin.ccrba` → `data.Ig`)을
   지평 전체에 대해 상수로 사용.
3. **자이로스코픽 항 무시**: $\omega \times I\omega$ 생략.
4. **발 lever arm $r_i = p_{foot,i} - p_{COM}$ 사전 계산**: stance 발은 현재 측정값,
   swing 발의 다음 착지는 Raibert 휴리스틱으로 예측 (아래 §5) — 지평 내에서 상수 취급.

이 근사들 덕분에 동역학이 상태에 대해 선형이 되고, 최적화가 QP로 풀린다.

연속시간 행렬 (`ComTraj._continuousDynamics`):

$$
A_c=\begin{bmatrix}0&0&I_3&0\\0&0&0&R_z^T\\0&0&0&0\\0&0&0&0\end{bmatrix},\quad
B_c^{(i)}=\begin{bmatrix}0&\cdots&0\\0&\cdots&0\\ \tfrac{1}{m}I_3&\cdots&\tfrac{1}{m}I_3\\ I^{-1}[r_1]_\times&\cdots&I^{-1}[r_4]_\times\end{bmatrix},\quad
g_c=[0;0;0;\,0;0;0;\,0;0;-9.81;\,0;0;0]
$$

---

## 3. 이산화 (`ComTraj._discreteDynamics`)

$dt$ = `MPC_DT`에 대해 **2차 정확도 명시적 이산화**를 사용한다 (지수행렬보다 빠르고,
2026-02 업데이트로 반복당 ~1.57배 속도 향상):

$$
x_{k+1} = A_d\,x_k + B_d^{(k)}\,u_k + g_d
$$

- $A_d = I_{12}$ + ( $p \leftarrow v\,dt$ ), ( $rpy \leftarrow R_z^T\,\omega\,dt$ )
- $g_d$: $p \mathrel{+}= \tfrac{1}{2}g\,dt^2$, $v \mathrel{+}= g\,dt$
- $B_d^{(k)}$: 힘 → 속도 ($\tfrac{dt}{m}$), 힘 → 위치 ($\tfrac{dt^2}{2m}$),
  힘 → 각속도 ($dt\,I^{-1}[r_i]_\times$), 힘 → 자세 ($\tfrac{dt^2}{2}R_z^T I^{-1}[r_i]_\times$, 2차 항)

$B_d^{(k)}$는 발 lever arm이 스텝마다 달라지므로 시변 (N개의 12×12 블록).

---

## 4. QP 공식화 (`centroidal_mpc.py`)

결정변수는 상태와 입력을 모두 포함 (**동시최적화**, condensing 없음):

$$
w = [\,x_1,\dots,x_N,\ u_0,\dots,u_{N-1}\,] \in \mathbb{R}^{24N} \quad (N=16 \Rightarrow 384)
$$

### 비용 함수

$$
\min_w \sum_{k=1}^{N} \|x_k - x_k^{ref}\|_Q^2 \;+\; \sum_{k=0}^{N-1}\|u_k\|_R^2
$$

- 기본 $Q = \mathrm{diag}(1,1,50,\ 10,20,1,\ 2,2,1,\ 1,1,1)$ — trot 보행용
- $R = 10^{-5} I_{12}$ — 힘 크기 정규화 (발 사이 힘 분배를 결정)
- 예제별 오버라이드 가능: `CentroidalMPC(go2, traj, Q=..., R=...)`
  (예: 스탠딩 자세제어는 위치/자세 가중치를 크게 — `ex05`, `ex06` 참조)

Hessian은 상수이므로 초기화 때 한 번만 조립 (`H_const`).

### 제약 조건

**1) 동역학 등식** — 각 스텝에 대해 $x_{k+1} - A_d x_k - B_d u_k = g_d$:

$$
\underbrace{\begin{bmatrix} I & & & \\ -A_d & I & & \\ & \ddots & \ddots & \\ & & -A_d & I \end{bmatrix}}_{I + S\otimes(-A_d)} X
\;-\; \mathrm{blkdiag}(B_d^{(0)},\dots,B_d^{(N-1)})\, U
= \begin{bmatrix} A_d x_0 + g_d \\ g_d \\ \vdots \\ g_d \end{bmatrix}
$$

($x_0$은 현재 측정 상태 — 매 solve마다 우변만 갱신)

**2) 마찰 피라미드 (부등식)** — stance 발마다 4면:

$$
|f_x| \le \mu f_z, \qquad |f_y| \le \mu f_z \qquad (\mu = 0.8)
$$

계수 행렬은 상수라 초기화 때 조립 (`_precompute_friction_matrix`), 상한(0 또는 ∞)만
접촉 테이블에 따라 매 solve마다 스위칭.

**3) 박스 제약** (`_compute_bounds`):
- **swing 발**: $f = 0$ (세 성분 모두)
- **stance 발**: $f_z \ge 10\,\mathrm{N}$ (미끄럼/조기 이륙 방지)

### 솔버

- **OSQP** (CasADi `conic` 인터페이스), 희소 구조 고정 → 심볼릭 분해 재사용
- **웜스타트**: 이전 solve의 primal/dual 해 재사용
- 측정 성능: 행렬 갱신 ~1 ms + solve ~1.7 ms (48 Hz 예산 20.8 ms 대비 여유)
- 첫 GRF $u_0$만 다리 제어기로 전달 (receding horizon)

---

## 5. 기준 궤적 생성 (`com_trajectory.py`)

명령 인터페이스 (`generate_traj`):

| 명령 | 기준 궤적 반영 |
|---|---|
| $v_x, v_y$ (body) | $R_z$로 월드 변환, $p^{ref}(t) = p_{des} + v\,t$, $v^{ref} = v$ |
| $z$ 높이 | $p_z^{ref}$ 상수 + 수직 속도 피드포워드 `z_vel_des_body` |
| roll/pitch (+rate FF) | $\phi^{ref}(t) = \phi_{des} + \dot\phi_{des}\,t$, $\omega^{ref}_{x,y} = \dot\phi_{des}$ |
| yaw rate | $\psi^{ref}(t) = \psi_0 + \dot\psi_{des}\,t$, $\omega^{ref}_z = \dot\psi_{des}$ |
| $x,y$ 절대 위치 (`x_pos_des_world`) | 스탠딩 body sway용 직접 위치 명령 |

- 위치 드리프트 방지: desired 위치는 현재 COM에서 **±0.1 m로 클램프**
- 속도/각속도 피드포워드는 사인 명령 추종 시 위상 지연을 없애는 핵심 (스탠딩 자세제어에서 검증)
- 지평 내 발 lever arm: 접촉 테이블을 순회하며 stance 구간은 값 유지, swing→touchdown
  전환 시 **Raibert 휴리스틱**으로 착지점 예측:

$$
p_{td} = p_{hip} + v\,\frac{T_{pred}}{2} + k_p(p - p_{des}) + k_v(v - v_{des}) + \Delta_{yaw}
$$

---

## 6. 게이트 스케줄러 (`gait.py`)

위상 기반 접촉 테이블: 다리 $i$의 위상 $\varphi_i(t) = \mathrm{mod}(\varphi_i^0 + t/T,\ 1)$,
$\varphi_i < \mathrm{duty}$ 이면 stance.

| 게이트 | 위상 오프셋 (FL,FR,RL,RR) | 비고 |
|---|---|---|
| **Trot** (`Gait`) | (0.5, 0.0, 0.0, 0.5) | 대각 2발 교대. 검증: 3 Hz, duty 0.6 |
| **Stand** (`StandGait`) | — (항상 전 발 접촉) | 스탠딩 몸통 자세제어용. 주파수는 MPC 지평 길이만 결정 |

Swing 발 궤적: 최소저크 5차 다항식 + 부드러운 z-범프 $64s^3(1-s)^3$ (정점 높이 0.1 m).

---

## 7. 다리 제어기 (`leg_controller.py`, 200 Hz)

### Stance 다리 — MPC 접촉력 실현

$$
\tau = J^T(-f_{MPC}) \;+\; \underbrace{(C\dot q + g)_{leg}}_{\text{다리 중력/코리올리 보상}} \;+\; \underbrace{\tau_{fric}\tanh(\dot q_j / 0.02)}_{\text{관절 건마찰 보상}}
$$

- $J$: 발 위치 3×3 자코비안 (월드, 해당 다리 관절만)
- **중력/코리올리 보상**: 이것이 없으면 실현되는 접촉력이 다리 링크 무게만큼 편향되어
  느린 위치 명령(스탠딩 sway)에서 몸통이 목표에 못 미침
- **건마찰 보상**: MJCF `frictionloss`에 대응 (로봇별 상수, 아래 표). 마찰 데드밴드가
  작은 힘 보정을 흡수하는 것을 방지

### Swing 다리 — 궤적 추종 임피던스 제어

$$
F = K_p(p_{des} - p) + K_d(v_{des} - v) + \Lambda(\ddot p_{des} - \dot J\dot q), \qquad
\tau = J^T F + (C\dot q + g)_{leg}
$$

- $K_p = 400 I$, $K_d = 75 I$, $\Lambda = (J M^{-1} J^T)^{-1}$: 조작공간 관성 (동역학 피드포워드)

토크는 예제 레벨에서 로봇별 한계로 포화 후 인가.

---

## 8. 로봇 모델 (`go2_robot_data.py`, `mujoco_model.py`)

로봇별 파라미터는 Pinocchio 모델 클래스의 **클래스 속성**으로 정의된다:

| 파라미터 | Go2 (`PinGo2Model`) | A2 (`PinA2Model`) |
|---|---|---|
| 총질량 | ~15 kg | ~40 kg |
| 베이스 프레임 | `base` | `base_link` |
| 기본 스탠스 (hip, thigh, calf) | (0, 0.9, −1.8) rad | (0, 0.8, −1.6) rad |
| 기본 베이스 높이 | 0.27 m | 0.415 m |
| 발 반지름 (터치다운 z) | 0.02 m | 0.032 m |
| 관절 건마찰 (hip/thigh/calf) | 0.2 / 0.2 / 0.2 Nm | 1.0 / 1.0 / 3.0 Nm |
| 토크 한계 (hip/thigh/calf) | 23.7 / 23.7 / 45.43 Nm | 120 / 120 / 180 Nm |

- 질량/관성텐서는 URDF에서 Pinocchio가 자동 산출 — MPC에 하드코딩 없음
- MuJoCo↔Pinocchio 상태 동기화는 **관절 이름 기반 매핑** (A2는 MJCF 다리 선언 순서가
  FL,RL,FR,RR로 Pinocchio 순서 FL,FR,RL,RR와 다름)
- 새 로봇 추가 = URDF/MJCF를 `models/`에 넣고 두 클래스 상속 + 속성 정의

---

## 9. 튜닝 노트

**$Q$ 대각 원소의 의미** (순서: $p_{x,y,z}$, rpy, $v_{x,y,z}$, $\omega$):
- 위치/자세 가중치 = 강성(P게인 유사), 속도 가중치 = 감쇠(D게인 유사)
- **위치 가중치만 키우면 저감쇠 발진** → 속도 가중치를 함께 상향
  (스탠딩 y-sway에서 Q=100/속도2로 전도 사례 → 200/40으로 안정화)
- 무거운 로봇(A2)은 같은 오차에 더 큰 힘·운동량 → 감쇠 가중치 상향 필요 (ex06: 80/80/5)

**대역폭 한계**:
- MPC 지평(~0.33 s)이 기준 궤적을 선형 외삽하므로 빠른 사인 명령의 피크가 깎임
  → x/y sway는 0.25 Hz 사용 (자세는 0.5 Hz 가능)
- 관절 건마찰은 잔류 정적 오차를 만든다 (적분기 없음). 보상 피드포워드가 완화하지만
  속도≈0 근처(피크)에서는 원리적으로 완전 제거 불가

**마찰 계수**: MPC의 $\mu = 0.8$은 MuJoCo 접촉 설정과 일치해야 함.

---

## 10. 코드 맵

| 파일 | 내용 |
|---|---|
| `src/convex_mpc/centroidal_mpc.py` | QP 조립/갱신/solve (CasADi + OSQP), 비용/제약/웜스타트 |
| `src/convex_mpc/com_trajectory.py` | 기준 궤적, 접촉 테이블 연동, 발 lever arm, $A_d/B_d/g_d$ 이산화 |
| `src/convex_mpc/gait.py` | `Gait`(trot), `StandGait`, Raibert 착지점, swing 궤적 |
| `src/convex_mpc/leg_controller.py` | stance/swing 토크, 중력·코리올리·건마찰 보상 |
| `src/convex_mpc/go2_robot_data.py` | Pinocchio 모델 (`PinGo2Model`, `PinA2Model`), 상태/기구학/동역학 항 |
| `src/convex_mpc/mujoco_model.py` | MuJoCo 시뮬 래퍼, Pinocchio 상태 동기화, 리플레이 뷰어 |
| `examples/ex00–ex04` | Go2 trot 데모 (제자리/전진/횡보행/회전) |
| `examples/ex05` | Go2 스탠딩 자세제어 (roll/pitch/yaw/높이/x·y sway) |
| `examples/ex06` | A2 스탠딩 자세제어 |
