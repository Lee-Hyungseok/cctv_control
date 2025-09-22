# CCTV Q-Learning Reinforcement Learning System

사거리 CCTV 범죄 탐지를 위한 Q러닝 강화학습 시스템

## 개요

이 프로젝트는 사거리에 설치된 CCTV가 Q러닝 강화학습을 통해 범죄 장면을 효율적으로 탐지하는 시스템을 구현합니다. 기존의 순차적 방향 촬영 방식과 비교하여 성능을 평가합니다.

## 시스템 구성

### 환경 (Environment)
- **위치**: 사거리 교차점
- **방향**: 동서남북 4개 방향
- **범죄 발생**: 각 방향에서 랜덤으로 발생
- **시뮬레이션 기간**: 365일 (525,600분)

### 에이전트 (Agent)
- **알고리즘**: Q-Learning
- **액션**: 4개 방향 중 하나 선택 (North, South, East, West)
- **상태**: 각 방향의 범죄 발생 상황
- **리워드**: 범죄 탐지 시 +10

### 기준 시스템 (Baseline)
- **방식**: 순차적 방향 촬영
- **패턴**: 각 방향을 1분씩 순차적으로 모니터링

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
# 완전한 시뮬레이션 (100 에피소드 + 10분 동영상)
python integrated_simulation.py

# 빠른 테스트 (50 에피소드, 동영상 없음)
python integrated_simulation.py --quick

# 동영상 없이 실행
python integrated_simulation.py --no-video

# 에피소드 수 조정
python integrated_simulation.py --episodes 200
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
- `comprehensive_performance_analysis.png`: 상세 성능 분석 그래프
- `intersection_comparison.png`: 교차점 시각화 비교
- `test_results.png`: 기본 테스트 결과
- `evaluation_report.txt`: 종합 평가 보고서

## 실험 설정

### Q-Learning 파라미터
- **학습률 (Learning Rate)**: 0.1
- **할인인자 (Discount Factor)**: 0.95
- **초기 Epsilon**: 1.0
- **Epsilon 감소율**: 0.995
- **최소 Epsilon**: 0.01

### 환경 설정
- **범죄 발생 확률**: 0.05 (5%)
- **훈련 에피소드**: 100회
- **평가 기간**: 365일 (525,600분)

## 예상 결과

Q-Learning 시스템은 다음과 같은 개선을 보일 것으로 예상됩니다:

1. **적응적 모니터링**: 범죄 패턴에 따른 동적 방향 선택
2. **향상된 탐지율**: 순차적 방식 대비 효율적인 범죄 탐지
3. **학습 기반 최적화**: 시간에 따른 성능 개선

## 기술적 특징

### Q-Learning 구현
- 상태-액션 테이블 기반 학습
- Epsilon-greedy 탐험 전략
- 경험 기반 정책 개선

### 시각화
- 실시간 교차점 모니터링 시각화
- 성능 지표 그래프
- 훈련 진행 상황 추적

## 향후 개선 방안

1. **실제 데이터 활용**: 실제 범죄 패턴 데이터로 학습
2. **딥러닝 확장**: DQN (Deep Q-Network) 적용
3. **다중 에이전트**: 여러 CCTV 간 협력 시스템
4. **실시간 적용**: 실제 CCTV 시스템과 연동

## 개발자 정보

이 시스템은 강화학습을 활용한 스마트 보안 시스템 연구의 일환으로 개발되었습니다.

---

**참고**: 이 시스템은 연구 목적으로 개발되었으며, 실제 보안 시스템 적용 시에는 추가적인 검증과 최적화가 필요합니다.