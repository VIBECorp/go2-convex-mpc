# 실로봇을 위한 상태추정 (Leg Odometry + IMU Kalman Filter)

시뮬레이션에서는 MPC 상태 $x$의 COM 위치·속도를 MuJoCo ground truth에서 직접 읽는다
(`mujoco_model.py :: update_pin_with_mujoco`). 실제 로봇에는 ground truth가 없으므로
**고유수용성(proprioceptive) 센서만으로 이를 추정**해야 한다. 이 문서는 그 방법과
본 저장소의 구현(`src/convex_mpc/state_estimator.py`), 그리고 시뮬레이션 검증 결과
(`examples/ex07_state_estimation.py`)를 정리한다.

---

## 1. 문제 정의

| 상태 | 시뮬레이션 | 실제 로봇 |
|---|---|---|
| 자세 (roll/pitch/yaw), 각속도 | ground truth | **IMU** (자세 융합 출력 + 자이로) — 해결됨 |
| 베이스/COM **위치, 속도** | ground truth | **직접 측정 불가 → 추정 필요** |
| 관절 위치/속도 | ground truth | 엔코더 (정밀, 문제없음) |

GPS는 실내 불가·저정밀, 비전/LiDAR는 별도 인프라와 지연이 필요하다. 사족보행 로봇의
표준 해법은 **다리 오도메트리(leg odometry)** 다.

## 2. 핵심 아이디어: 접지된 발은 움직이지 않는다

Stance 발이 미끄러지지 않는다면, 그 발은 위치·속도의 "측정치" 역할을 한다.
IMU 자세 $R_{wb}$와 엔코더 $(q, \dot q)$로부터:

- **발의 베이스 기준 위치** (순기구학): $p_{f}^{b} = fk(q)$
- **월드 기준 발 속도**:

$$
v_{f}^{w} = v + R_{wb}\,(\omega_b \times p_f^b + J(q)\,\dot q) \;\overset{stance}{=}\; 0
\quad\Rightarrow\quad
v = -R_{wb}\,(\omega_b \times p_f^b + J\dot q)
$$

접지 발 하나마다 베이스 속도 측정치가 하나씩 나오고, 여러 발을 칼만필터로 융합한다.

## 3. 선형 칼만필터 공식 (MIT Cheetah 3 방식)

자세는 IMU(또는 상위 자세 필터)에서 받아 **확정된 것으로 취급**한다. 그러면 남는
추정 문제가 완전히 **선형**이 되어 EKF가 아닌 일반 KF로 풀린다.

**상태 (18차원):**

$$
\hat x = [\;p\ (3),\quad v\ (3),\quad p_{f,1..4}\ (12)\;] \qquad \text{(모두 월드 프레임)}
$$

**예측 (IMU strapdown 적분):**

$$
a_w = R_{wb}\,a_{IMU} + g,\qquad
p \leftarrow p + v\,dt + \tfrac{1}{2}a_w dt^2,\qquad
v \leftarrow v + a_w dt,\qquad
p_{f,i} \leftarrow p_{f,i}
$$

발 위치의 프로세스 노이즈는 stance일 땐 작게(미세 슬립), **swing일 땐 크게** 주어
착지 시 기구학 값으로 빠르게 재수렴하게 한다.

**보정 (엔코더 기구학, 모두 상태에 선형):**

| # | 측정 모델 | 적용 대상 |
|---|---|---|
| 1 | $p_{f,i} - p = R_{wb}\,fk_i(q)$ (상대 발 위치) | 모든 발 (기구학은 항상 유효) |
| 2 | $v = -R_{wb}(\omega_b \times fk_i + J_i \dot q)$ (정지 발 가정) | stance 발만 |
| 3 | $p_{f,i,z} = r_{foot}$ (평지 가정) | stance 발만 |

Swing 발의 2·3번 행은 제거하는 대신 **측정 노이즈를 크게 부풀려** 무시한다
(행렬 크기 고정 → 구현 단순).

**접촉 판별:** 게이트 스케줄의 예상 접촉을 사용하되, **접촉 전환 직전·직후
±30 ms는 신뢰하지 않는다** (실제 착지/이륙이 스케줄보다 늦거나 빨라 발이 아직
움직이는 구간의 속도 측정이 편향을 만든다 — ex07에서 드리프트 9.5%→7.3% 개선 확인).
실로봇에서는 관절 토크로 추정한 발 힘($f = -J^{-T}\tau$) 또는 발 힘센서를 함께 쓰면
더 좋다.

## 4. 관측성: 무엇이 추정되고 무엇이 드리프트하는가

| 상태 | 관측성 | 비고 |
|---|---|---|
| roll, pitch | 관측 가능 (중력 방향) | IMU에서 직접 |
| 속도 $v$ | **관측 가능** (다리 오도메트리) | 제어에 가장 중요, 드리프트 없음 |
| 높이 $z$ | 접지면 기준 관측 가능 | 평지 가정 하에 정확 |
| 절대 $x, y$ (및 yaw) | **관측 불가 → 서서히 드리프트** | 아래 참조: 이 MPC에는 무해 |

**절대 x, y 드리프트가 이 컨트롤러에 무해한 이유:**

1. `com_trajectory.py`의 desired 위치는 현재 추정 위치 기준 **±0.1 m로 클램프**되어
   위치 피드백이 항상 "최근 위치 대비 상대적"으로만 작동한다.
2. MPC 동역학의 발 lever arm $r_i = p_{foot} - p_{COM}$은 상대량이라 절대 드리프트가
   소거된다 (발 위치와 베이스를 **같은 추정기에서 일관되게** 뽑는 것이 조건 —
   KF 상태에 발 위치가 포함된 이유이기도 하다).
3. 스탠딩 x/y sway(ex05/ex06)의 기준점 `x_ref0`도 추정기 출력에서 캡처하는 상대 명령이다.

절대 위치가 실제로 필요한 것은 웨이포인트 주행 같은 **상위 작업**뿐이며, 그때만
VIO/LiDAR-SLAM/모션캡처/GPS를 낮은 주기로 융합해 드리프트를 보정하면 된다.

## 5. 구현: `state_estimator.py :: LegOdometryKF`

```python
from convex_mpc.state_estimator import LegOdometryKF

est = LegOdometryKF(robot, dt)                 # robot: PinGo2Model 또는 PinA2Model
est.reset(p0, R_wb, q_joints)                  # 시작 자세에서 초기화 (오도메트리 원점)
est.update(R_wb, gyro_body, accel_body,        # 매 제어 틱 (200 Hz)
           q_joints, dq_joints, contact_mask)
est.base_pos, est.base_vel, est.foot_pos       # 추정치
```

- 기구학은 전용 Pinocchio 모델 인스턴스(베이스를 원점에 고정)로 계산 —
  로봇 클래스(`PinGo2Model`/`PinA2Model`)를 그대로 재사용하므로 Go2/A2 모두 동작
- $H$ 행렬은 상수, 접촉에 따라 노이즈 공분산 $R$만 스위칭
- 18×18 KF + 기구학 갱신으로 200 Hz에서 여유 있게 동작
- 가속도계 바이어스는 상태에 포함하지 않음 — 속도 측정이 지속 보정하므로 정상 동작
  하지만, 장시간 운용 시 바이어스 상태(+3차원) 추가가 정석 확장

**COM 상태로의 변환:** 추정기는 베이스 프레임의 pose/twist를 내놓는다. MPC가 쓰는
COM 위치/속도는 (베이스 상태 + 엔코더) configuration으로 Pinocchio가 계산하므로
별도 추정이 필요 없다. 즉 실로봇 이식 시 교체 지점은
`update_pin_with_mujoco()`가 하던 일(q_pin, dq_pin 채우기) **한 곳**이다:

```
q_pin  = [ est.base_pos, quat(IMU), q_encoder(12) ]
dq_pin = [ R_wb.T @ est.base_vel, gyro_body, dq_encoder(12) ]   # 선속도는 body frame
```

## 6. 시뮬레이션 검증 (`examples/ex07_state_estimation.py`)

Trot 보행(전진 0.5 m/s → 제자리 → 전진+회전, 6 s) 중 KF가 **노이즈를 입힌
IMU + 엔코더 신호만으로** 추정하고 ground truth와 비교한다. 파일 상단의
`USE_ESTIMATOR` 플래그로 두 모드를 선택할 수 있다:

- **`True` (기본, 폐루프)**: MPC/다리 제어기가 **KF 추정치 + 노이즈 엔코더**로
  구동된다 — 실제 로봇과 동일한 구성. ground truth는 평가에만 사용.
- **`False` (개루프)**: 컨트롤러는 ground truth로 구동되고 KF는 병렬 평가만.

주입한 센서 노이즈: 자이로 0.01 rad/s, 가속도계 0.1 m/s² + **상수 바이어스
(0.05, −0.03, 0.08) m/s²**, IMU 자세 오차 0.002 rad, 엔코더 0.001 rad / 0.01 rad/s.

**결과:**

| 항목 | 폐루프 (KF로 제어) | 개루프 (참고) |
|---|---|---|
| 보행 안정성 | 전도 없음, 높이 0.27–0.30 m 유지, 2.4 m 주행 | 동일 |
| 속도 RMS 오차 | $v_x$ 0.032, $v_y$ 0.019, $v_z$ 0.028 m/s | 0.032 / 0.016 / 0.024 |
| 높이 RMS 오차 | 5.2 mm | 4.4 mm |
| xy 드리프트 | 0.18 m (주행거리의 6.6%) | 0.18 m (7.3%) |

**폐루프에서도 추정 품질과 보행 성능이 개루프와 사실상 동일**하다 — 추정 오차가
제어 루프를 거쳐 증폭되지 않음을 보여준다 (§4의 구조적 이유: 위치 피드백이 상대적,
속도는 드리프트 없이 관측됨).

**드리프트 원인 분석:** 같은 시뮬레이션에서 ground truth로 잰 stance 발의 실제 누적
xy 슬립이 **다리당 ~0.08 m**였다. 즉 드리프트의 지배 요인은 필터 결함이 아니라
"발이 고정"이라는 가정 자체가 물리적으로 깨지는 것(trot 중 미세 슬립)이며, 이는
실제 로봇에서도 동일하게 나타나는 다리 오도메트리의 원리적 한계다. §4에서 설명했듯
이 드리프트는 본 컨트롤러에 무해하다.

## 7. 실로봇 적용 시 체크리스트

1. **1차 구현**: Unitree SDK(lowstate/sportmode)의 자체 융합 속도·위치 추정치를
   그대로 사용 — 가장 빠른 경로. 정밀도가 아쉬우면 본 KF로 교체.
2. **접촉 판별 강화**: 스케줄 + 발 힘 추정(관절 토크 기반) 또는 A2/Go2 발 센서로
   게이팅. 착지 충격 직후에는 측정 노이즈를 일시 상향.
3. **IMU 장착 오프셋**: 가속도계가 베이스 원점에서 벗어나 있으면
   $\omega \times (\omega \times r)$ 원심 항 보정 (Go2 IMU는 ~5 cm 오프셋, 영향 작음).
4. **바이어스 상태 추가**: 장시간 운용 시 가속도계 바이어스 3차원을 상태에 포함.
5. **비평지 지형**: 발 높이 측정(#3)의 평지 가정을 지형 추정으로 대체하거나 제거.
6. **절대 위치가 필요한 작업**: VIO/LiDAR 오도메트리를 저주기 보정으로 융합.

## 참고문헌

- Bledt et al., *"MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot"*, IROS 2018 — 본 KF의 원형
- Bloesch et al., *"State Estimation for Legged Robots – Consistent Fusion of Leg Kinematics and IMU"*, RSS 2012 — 관측성 분석 (절대 위치·yaw 비관측성 증명)
