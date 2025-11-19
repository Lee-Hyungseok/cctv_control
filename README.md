# CCTV Q-Learning Reinforcement Learning System

사거리 CCTV 범죄 탐지를 위한 Q러닝 강화학습 시스템

## 개요

이 프로젝트는 사거리에 설치된 CCTV가 Q러닝 강화학습을 통해 범죄 장면을 효율적으로 탐지하는 시스템을 구현합니다. 기존의 순차적 방향 촬영 방식과 비교하여 성능을 평가합니다.

## 시스템 구성

### 환경 (Environment)
- **위치**: 사거리 교차점
- **방향**: 동서남북 4개 방향
- **범죄 발생 확률**: 각 방향에서 5% 확률로 랜덤 발생
- **에피소드 길이**: 144 스텝 (하루 = 10분 단위)
- **훈련 기간**: 1000 에피소드 (1000일)
- **평가 기간**: 365 에피소드 (365일)

### 에이전트 (Agent)
- **알고리즘**: Q-Learning (Tabular)
- **액션**: 4개 방향 중 하나 선택 (North, South, East, West)
- **상태**: 현재 방향, 범죄 감지 여부, 시간 단계 (10분 단위 bin)
- **리워드**:
  - 범죄 탐지 시: +10
  - 범죄 미탐지 시: -1 (놓친 범죄가 있을 때)
- **탐험 전략**: Epsilon-greedy (epsilon=0.5 고정, 50% 랜덤 탐색)

### 기준 시스템 (Baseline)
- **방식**: 순차적 방향 촬영 (Sequential)
- **패턴**: North → South → East → West 순서로 순환

## 파일 구조

```
cctv_control/
├── cctv_environment.py          # 환경 클래스
├── q_learning_agent.py          # Q러닝 에이전트
├── baseline_cctv.py             # 기준 순차 시스템
├── main_simulation.py           # 메인 시뮬레이션 (훈련 + 동영상)
├── report_generator.py          # 보고서 생성기
├── integrated_simulation.py     # **통합 워크플로우 (메인 실행 파일)**
├── quick_test.py               # 통합 테스트
├── requirements.txt            # 필요 패키지
└── README.md                   # 문서
```

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 통합 시뮬레이션 실행 (권장)
```bash
# 완전한 시뮬레이션 (1000 훈련 에피소드 + 365 평가 에피소드 + 2분 동영상)
# 예상 소요 시간: 약 20분
python integrated_simulation.py

# 빠른 테스트 (100 훈련 에피소드, 10 평가 에피소드, 동영상 없음)
# 예상 소요 시간: 약 1분
python integrated_simulation.py --quick

# 동영상 없이 실행 (시간 절약)
python integrated_simulation.py --no-video

# 훈련 에피소드 수 조정
python integrated_simulation.py --episodes 2000

# 평가 에피소드 수 조정
python integrated_simulation.py --episodes 1000 --eval-episodes 180
```

### 3. 개별 모듈 실행
```bash
# 통합 테스트 (시스템 검증)
python quick_test.py

# 메인 시뮬레이션만 (훈련 + 동영상만)
python main_simulation.py
```

## 결과 분석

### 성능 지표
1. **탐지율 (Detection Rate)**: 발생한 범죄 중 탐지한 비율
2. **총 리워드 (Total Reward)**: 365일간 획득한 총 점수
3. **탐지된 범죄 수**: 실제 탐지한 범죄 건수

### 생성되는 파일
- `cctv_simulation.mp4`: 2분 분량 애니메이션 (마지막 30일 시각화)
- `training_results.png`: 훈련 과정 분석 (누적 리워드, 누적 평균 탐지율 포함)
- `evaluation_results.png`: 평가 단계 성능 비교
- `comprehensive_performance_analysis.png`: 9-패널 상세 성능 분석
- `intersection_comparison.png`: 교차점 시각화 비교
- `evaluation_report.txt`: 종합 평가 보고서 (방향별 통계 포함)

## 실험 설정

### Q-Learning 파라미터
- **학습률 (Learning Rate)**: 0.1
- **할인인자 (Discount Factor)**: 0.95
- **Epsilon (고정)**: 0.5 (50% 탐험, decay 없음)
- **Epsilon Decay**: 1.0 (고정 전략)
- **최소 Epsilon**: 0.5 (고정)

### 환경 설정
- **범죄 발생 확률**: 0.05 (5% per step per direction)
- **훈련 에피소드**: 1000회 (기본값)
- **평가 에피소드**: 365일 (기본값)
- **에피소드당 스텝**: 144 (하루 = 10분 × 144)

## 실험 결과

### 주요 발견사항

Q-Learning 시스템은 다음과 같은 특성을 보입니다:

1. **탐험과 활용의 균형**: Epsilon=0.5로 설정하여 50% 탐험을 유지
   - 다양한 방향의 범죄를 놓치지 않고 탐지
   - 학습된 정책과 랜덤 탐색의 균형

2. **방향별 탐지율 분석**:
   - 각 방향(동서남북)별 범죄 발생 및 탐지 통계 제공
   - Q-Learning과 Baseline의 방향별 성능 비교

3. **누적 성능 추적**:
   - 누적 리워드를 통한 장기 성능 추세 분석
   - 누적 평균 탐지율로 학습 안정성 확인

4. **동일 환경 비교**:
   - Q-Learning과 Baseline이 동일한 범죄 환경(seed)에서 평가
   - 공정한 성능 비교 보장

## 기술적 특징

### Q-Learning 구현
- **상태 표현**: defaultdict 기반 동적 Q-table
  - 상태 키: `{current_direction}_{crime_detected}_{time_bin}`
  - 시간 bin: 10분 단위로 그룹화
- **Epsilon-greedy 전략**: 고정 epsilon=0.5 (decay 없음)
- **TD Learning**: 시간차 학습으로 Q값 업데이트
- **동일 환경 보장**: Seed 기반 재현 가능한 범죄 생성

### 시각화
- **훈련 결과**: 6-패널 그래프
  - 누적 리워드 비교 (Q-Learning vs Baseline)
  - 누적 평균 탐지율 비교
  - 이동 평균 스무딩
  - Epsilon 변화 추적
- **평가 결과**: 에피소드별 리워드 비교
- **애니메이션**: 2분 분량 실시간 모니터링 (마지막 30일)
- **종합 분석**: 9-패널 상세 성능 분석 및 방향별 통계

## 향후 개선 방안

1. **고급 강화학습 알고리즘**:
   - DQN (Deep Q-Network) 적용
   - Double DQN, Dueling DQN 등 변형 기법
   - PPO, A3C 등 Policy Gradient 방법

2. **상태 표현 개선**:
   - 범죄 발생 빈도 히스토리 추가
   - 시간대별 패턴 정보 통합
   - 이미지 기반 상태 표현 (CNN)

3. **실제 데이터 활용**:
   - 실제 범죄 패턴 데이터로 학습
   - 지역별, 시간대별 범죄 발생 통계 반영

4. **다중 에이전트 시스템**:
   - 여러 CCTV 간 협력 학습
   - 통신 프로토콜 설계

5. **실시간 적용**:
   - 실제 CCTV 시스템과 연동
   - 경량화 및 실시간 추론 최적화

## 개발자 정보

이 시스템은 강화학습을 활용한 스마트 보안 시스템 연구의 일환으로 개발되었습니다.

---

**참고**: 이 시스템은 연구 목적으로 개발되었으며, 실제 보안 시스템 적용 시에는 추가적인 검증과 최적화가 필요합니다.