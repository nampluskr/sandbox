### Guidelines for References:

References should be written to the following formats:
Authors should be listed surname first, followed by a comma and initials of given names. If there are two or more authors, write only the name of the author and mark it as "et al." Titles of cited articles should be written in full, with only the first word capitalized. Book titles are italic with all main words capitalized. Journal titles are italic and abbreviated according to common usage. Volume numbers are bold. The publisher and the year of publication are required for books cited. The year of publication (posting) should written in parentheses. Refer to the reference writing example of this template

Examples:
[1] True, H. L. et al. A yeast prion provides a mechanism for genetic variation and phenotypic diversity, _Nature_ **407**, 477–483 (2000)
[2] Schluter, D. _The Ecology of Adaptive Radiation_ (Oxford Univ. Press, 2000)
[3] Seo, H. et al. Stacked Color Image Sensor using Wavelength-SelectiveOrganic Photoconductive Films with Zinc-Oxide Thin FilmTransistors as a Signal Readout Circuit. Proc. SPIE 2010, 7536,753602. (2010)
[4] Engelberg, J. et al. Near-IR Wide Field-of-View Huygens Metalens forOutdoor Imaging Applications. arXiv: 1901.07331 (2019)


---

반영할 내용

(특징) 
- 정상 데이터 기반 비지도 학습
- 다양한 OLED Display Killer 패턴에 대해 2D 계측기의 촬상 XYZ 데이터 활용
- 측정 - 전처리 - 학습 - 추론 모두 Batch Job 으로 실행되어 실시간 이슈 없음
- 신규 Unknown 불량 또는 얼룩 사전 검출로 불량 유출 방지
- 엔지니어의 육안 검사시 개인간 편차 제거
- AI 불량 사전 검출로 제품간 화질 검사의 공수 절감

(초록) 디스플레이 화질 스펙 검증과 얼룩 불량 판단은 검사자의 시각을 이용한 구동화질 평가에서 최종 결정된다. 휘도, 온도, 주파수 등 다양한 구동 조건에서 수동 검사 방식으로 진행되어 시간과 노력이 많이 소요된다. 또한 검사자간 눈높이 차이에 의한 유출 Risk가 항시 존재한다. 특히, 정상 데이터가 압도적으로 많고 불량이 희소하며 모양·위치·수준이 예측 불가능해 단일 임계값 기반 판정이 불안정하다. 
본 연구는 정상 데이터만으로 학습
패턴에 의존하지 않는 실용적 기준선으로서, 다양한 콘텐츠와 휘도 조건에서의 OLED 화질 이상 검출에 효과적이다. 제안된 방법은 실제 평가에 즉시 적용 가능하며, 신뢰성 높은 품질 관리 시스템 구축에 기여할 것으로 기대된다.

(서론) 디스플레이의 화질 평가는 계측기를 이용한 물리적 특성 기반의 정량적 평가와 인간의 시각 인지 특성 기반의 주관적 평가로 나눌 수 있다. 이러한 객관적 화질 측정방법은 사람이 실제로 평가하는 화질과는 차이가 있다. 따라서, 객관적 화질 측정방법보다는 주관적 화질 측정방법이 화질 측정에 더욱 적합하다고 알려져 있다. 무엇보다 저계조 또는 저휘도에서의 얼룩은 구동 환경과 사용자에 따라 수준 차이뿐 아니라 존재 유무에 판단도 상이할 수 있다.
OLED의 구동 화질 Risk 검증 및 불량 개선을 위해 제품의 다양한 환경, 기능, 패턴(암실, 고온, 저온, Dimming, 주파수, 복합 Killer 패턴)에 대한 광특성 계측 및 목시 평가를 진행한다. 

Fig. 1. 구동 화질 검사용 패턴 이미지 삽입

패널 내 이상점은 정규 패턴의 일정 위치에 나타나는 예측 가능한 이상점과 불규칙한 형태로 임의의 위치에 나타나는 예측 불가능한 이상점이 있으며, 이중 많은 이상점은 저계조에서 발생하기 때문에 검사자간 얼룩 검출 편차로 인한 검증 시간 증가 및 불량 유출을 초래할 수 있다.

본 논문에서는 AI 모델을 통해 구동화질 이상 자동 검출 시스템을 개발에 대하여 다룬다. 2D 촬상 및 화질 검증 시간을 단축하고, 검출력을 높여 불량 유출을 최소화 하고자 하였다. 특히, 저계조 저휘도 영역에서의 얼룩 판단을 위한 특화된 데이터 전처리 방법 소개하고 다양한 이상탐지 알고리즘 성능 비교와 최적화 방법을 다룬다.

(결론)
본 연구는 정상 데이터만 존재하고 불량이 희소한 생산 환경에서 임계값 의존성을 줄이기 위한 새로운 검출 프레임워크를 제시하였다.
본 프레임워크는 산업 환경에서 신뢰성 높은 OLED 화질 이상 검출을 위한 견고한 출발점을 제공한다. 해석 가능성과 실용성을 동시에 확보한 본 접근법은 다양한 산업 응용으로 확장 가능하며, 지속적인 개선을 통해 더욱 강건하고 효율적인 품질 관리 시스템 구축에 기여할 것으로 기대된다.
